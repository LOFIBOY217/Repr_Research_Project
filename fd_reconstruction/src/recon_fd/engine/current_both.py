"""Exact finite-pool FD gradients using moment adjoints and chunked replay.

The real whitening reference remains detached, as in AdvFD. 'Exact' refers
to the fake-side full-pool gradient of that stop-gradient objective, up to
floating-point rounding; it is not a derivative through real whitening.
"""
import time
import torch
from recon_fd.data import sequential_loader
from recon_fd.objectives.current_both import checked_reconstruction
from recon_fd.objectives.statistics import RunningMoments
from recon_fd.objectives.whitening import real_whitened_frechet_distance
from .gradients import checked_grad_norm


def full_pool_backward(model, objective, phase):
    if phase not in {"D", "G"}:
        raise ValueError("Expected D or G phase")
    if model.training or objective.extractor.training:
        raise RuntimeError("Two-pass gradients require deterministic eval-mode models")
    pair = objective.current_pair(model)
    objective.validate_pair(pair)
    mean = pair.fake.mean.clone().requires_grad_(True)
    covariance = pair.fake.cov.clone().requires_grad_(True)
    fd = real_whitened_frechet_distance(pair.real.moments.mean, pair.real.moments.cov,
                                       mean, covariance, objective.config["whiten_eps"])
    loss = -fd if phase == "D" else objective.config["weight"] * fd / (fd.detach() + objective.config["norm_eps"])
    dmean, dcov = torch.autograd.grad(loss, (mean, covariance))
    if not torch.isfinite(dmean).all() or not torch.isfinite(dcov).all():
        raise FloatingPointError("Non-finite full-pool moment derivative")
    n = pair.fake.count
    observed = RunningMoments()
    pool = objective.real_reference.pool
    for batch in sequential_loader(pool, objective.microbatch_size, objective.real_reference.workers):
        objective.validate_pair(pair)
        images = batch["image"].to(mean.device)
        if phase == "D":
            with torch.no_grad():
                reconstruction = checked_reconstruction(model, images)
        else:
            reconstruction = checked_reconstruction(model, images)
        features = objective.extractor(reconstruction)
        observed.update(features)
        # dL/df_i = dL/dmu / N + (f_i-mu) (dL/dC + dL/dC^T)/(N-1).
        # No graph through these adjoints: backward below computes the VJP.
        gradient = dmean / n + (features.detach().double() - pair.fake.mean) @ (dcov + dcov.T) / (n - 1)
        features.backward(gradient.to(features.dtype))
        objective.gradient_feature_images.add_(len(images))
    replay = observed.moments(ddof=1)
    # Catch stochastic or stateful two-pass forwards before any optimizer step.
    if (replay.count != n or not torch.allclose(replay.mean, pair.fake.mean, rtol=1e-5, atol=1e-8)
            or not torch.allclose(replay.cov, pair.fake.cov, rtol=1e-5, atol=1e-8)):
        raise RuntimeError("Replayed features differ from current-pool statistics; no optimizer update allowed")
    objective.validate_pair(pair)
    return pair, fd.detach(), loss.detach()


def current_both_g_step(model, objective, optimizer, critic_optimizer, grad_clip, step):
    started = time.monotonic()
    # Eval disables stochastic layers/buffer updates, NOT parameter gradients.
    model.eval().requires_grad_(False)
    objective.train()
    optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    metrics = {}
    try:
        if objective.critic_due(step):
            objective.set_critic_trainable(True)
            for _ in range(objective.config["steps_per_update"]):
                critic_optimizer.zero_grad(set_to_none=True)
                pair, fd, _ = full_pool_backward(model, objective, "D")
                if any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError("D-step leaked gradients to E+D")
                parameters = objective.critic_parameters()
                if not any(p.grad is not None for p in parameters):
                    raise RuntimeError("No adversarial gradient")
                norm = checked_grad_norm(parameters, objective.config["grad_clip"])
                objective.validate_pair(pair)
                critic_optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError("Non-finite adversarial parameters")
                objective.critic_updates.add_(1)
                metrics.update(critic_fd=float(fd), critic_grad_norm=float(norm))
    finally:
        critic_optimizer.zero_grad(set_to_none=True)
        objective.set_critic_trainable(False)
        model.requires_grad_(True)
    pair, fd, loss = full_pool_backward(model, objective, "G")
    norms = {}
    for name, parameters in model.parameter_groups().items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError(f"No gradient reached {name}")
        norms[name] = float(torch.stack([g.detach().float().norm().square() for g in gradients]).sum().sqrt())
    if any(p.requires_grad or p.grad is not None for p in objective.extractor.parameters()):
        raise RuntimeError("Extractor not frozen/cleared for G-step")
    norm = checked_grad_norm(model.parameters(), grad_clip)
    objective.validate_pair(pair)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite reconstruction parameters")
    objective.generator_updates.add_(1)  # Invalidates fake cache, even if D was skipped.
    real, fake = objective.real_reference, objective.fake_reference
    metrics.update(loss=float(loss), adv_fd=float(fd), raw_fd={}, normalized_fd={},
                   static_enabled=False, grad_norm=float(norm), group_grad_norm=norms,
                   adv_schedule_step=step, adv_effective_weight=objective.config["weight"],
                   critic_updated=objective.critic_due(step), critic_updates=int(objective.critic_updates),
                   generator_updates=int(objective.generator_updates),
                   loss_psi_version=pair.psi_version, loss_generator_version=pair.generator_version,
                   real_feature_version=pair.real.feature_psi_version,
                   fake_feature_psi_version=int(fake.psi_version),
                   fake_feature_generator_version=int(fake.generator_version),
                   real_reference_current_psi=True, fake_statistics_historical=False,
                   real_reference_refreshes=int(real.refreshes), fake_reference_refreshes=int(fake.refreshes),
                   real_reference_feature_images=int(real.feature_images),
                   fake_reference_feature_images=int(fake.feature_images),
                   gradient_feature_images=int(objective.gradient_feature_images),
                   real_reference_pool_samples=len(real.pool), fake_reference_pool_samples=len(real.pool),
                   real_ema_updates=0, fake_ema_updates=0, statistics_updates={},
                   fake_cache_current_after_G=fake.current(int(objective.critic_updates), int(objective.generator_updates)),
                   gradient_estimator="full_pool_two_pass_chain_rule", tokenizer_mode="eval_with_grad",
                   feature_microbatch_size=objective.microbatch_size,
                   step_seconds=time.monotonic() - started)
    return metrics
