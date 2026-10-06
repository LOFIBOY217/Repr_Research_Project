"""C: one reconstruction, optional D update, then joint E+D update."""
import time
import torch
from .adversarial import critic_step
from .gradients import checked_grad_norm


def candidate_g_step(model, objective, optimizer, critic_optimizer, images, grad_clip, step):
    started = time.monotonic()
    model.train().requires_grad_(True)
    objective.train()
    objective.set_critic_trainable(False)
    optimizer.zero_grad(set_to_none=True)
    reconstruction = model(images)
    if (reconstruction.shape != images.shape or not torch.isfinite(reconstruction).all()
            or reconstruction.min().detach() < 0 or reconstruction.max().detach() > 1):
        raise ValueError("Invalid reconstruction")
    model.requires_grad_(False)
    try:
        metrics = critic_step(objective, critic_optimizer, images, reconstruction, step)
    finally:
        model.requires_grad_(True)
    # D increments critic_updates: this invalidates the real-statistics cache.
    dynamic = objective.dynamic(images, reconstruction, step)
    normalized = dynamic.fd / (dynamic.fd.detach() + objective.config["norm_eps"])
    loss = objective.config["weight"] * normalized
    static = objective.static(reconstruction) if objective.static is not None else None
    if static is not None:
        loss = loss + static.loss
    loss.backward()
    norms = {}
    for name, parameters in model.parameter_groups().items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError(f"No gradient reached {name}")
        norms[name] = float(torch.stack([g.detach().float().norm().square() for g in gradients]).sum().sqrt())
    if any(p.requires_grad or p.grad is not None for p in objective.extractor.parameters()):
        raise RuntimeError("Candidate extractor not frozen/cleared for G-step")
    if any(p.requires_grad or p.grad is not None for s in objective.spaces.values() for p in s.extractor.parameters()):
        raise RuntimeError("Static ablation extractor changed gradient scope")
    norm = checked_grad_norm(model.parameters(), grad_clip)
    objective.validate_pending(dynamic)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite reconstruction parameters")
    if static is not None:
        objective.static.commit(static)
    objective.commit_dynamic(dynamic)
    reference = objective.real_reference
    metrics.update(loss=float(loss.detach()), adv_fd=float(dynamic.fd.detach()),
                   adv_normalized_fd=float(normalized.detach()),
                   raw_fd=static.raw if static is not None else {},
                   normalized_fd=static.normalized if static is not None else {},
                   static_enabled=static is not None, grad_norm=float(norm), group_grad_norm=norms,
                   adv_schedule_step=step, adv_effective_weight=objective.config["weight"],
                   critic_updated=objective.critic_due(step), critic_updates=int(objective.critic_updates),
                   real_reference_mode=reference.mode,
                   real_reference_current_psi=reference.mode == "reencode_pool",
                   real_feature_version=dynamic.real.feature_psi_version,
                   loss_psi_version=dynamic.psi_version,
                   real_reference_refreshes=int(reference.refreshes),
                   real_reference_pool_samples=len(reference.pool),
                   real_reference_feature_images=int(reference.feature_images),
                   real_ema_updates=int(reference.ema.updates) if reference.ema is not None else 0,
                   fake_ema_updates=int(objective.fake_statistics.updates) if objective.fake_statistics is not None else 0,
                   fake_statistics_historical=objective.fake_statistics is not None,
                   fake_statistics_samples=len(images) if objective.fake_statistics is None else 0,
                   statistics_updates={k: int(v.statistics.updates) for k, v in objective.spaces.items()},
                   clipped_pixel_fraction=float(((reconstruction.detach() <= 0) | (reconstruction.detach() >= 1)).float().mean()),
                   step_seconds=time.monotonic() - started)
    return metrics
