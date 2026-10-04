"""Reconstruction adapter for the official AdvFD D-then-G implementation."""
import torch


def critic_step(objective, optimizer, real, reconstruction, step):
    """No generator graph; D maximizes UNNORMALIZED whitened FD."""
    metrics = {}
    if not objective.critic_due(step):
        return metrics
    objective.set_critic_trainable(True)
    try:
        for _ in range(objective.config["steps_per_update"]):
            optimizer.zero_grad(set_to_none=True)
            result = objective.dynamic(real.detach(), reconstruction.detach(), step)
            (-result.fd).backward()
            parameters = objective.critic_parameters()
            if not any(p.grad is not None for p in parameters):
                raise RuntimeError("No adversarial parameter received a gradient")
            norm = torch.nn.utils.clip_grad_norm_(parameters, objective.config["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("Non-finite adversarial parameters")
            objective.critic_updates.add_(1)
            metrics = {"critic_fd": float(result.fd.detach()), "critic_grad_norm": float(norm)}
    finally:
        optimizer.zero_grad(set_to_none=True)
        objective.set_critic_trainable(False)
    # EMA commits happen once using the G-side forwards, NOT every D forward.
    return metrics


def adversarial_g_step(model, objective, optimizer, critic_optimizer, images, grad_clip, step):
    # Like upstream args.current_step, `step` is completed G steps (zero-based).
    model.train().requires_grad_(True)
    objective.train()
    objective.set_critic_trainable(False)
    optimizer.zero_grad(set_to_none=True)
    reconstruction = model(images)
    if reconstruction.shape != images.shape or not torch.isfinite(reconstruction).all():
        raise ValueError("Invalid reconstruction")
    if reconstruction.min().detach() < 0 or reconstruction.max().detach() > 1:
        raise ValueError("Reconstruction outside [0,1]")
    metrics = {}
    if objective.active(step):
        objective.initialize_fake_at_activation()
        model.requires_grad_(False)
        try:
            metrics.update(critic_step(objective, critic_optimizer, images, reconstruction, step))
        finally:
            model.requires_grad_(True)
    static = objective(reconstruction)
    loss = static.loss
    dynamic = None
    effective_weight = objective.effective_weight(step)
    if objective.active(step):
        # Recompute psi features after the D update, without regenerating x_hat.
        dynamic = objective.dynamic(images, reconstruction, step)
        normalized = dynamic.fd / (dynamic.fd.detach() + objective.static.norm_eps)
        loss = loss + effective_weight * normalized
        metrics.update(adv_fd=float(dynamic.fd.detach()), adv_normalized_fd=float(normalized.detach()))
    loss.backward()
    norms = {}
    for name, parameters in model.parameter_groups().items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError(f"No gradient reached {name}")
        norms[name] = float(torch.stack([g.detach().float().norm().square() for g in gradients]).sum().sqrt())
    if any(p.requires_grad or p.grad is not None for p in objective.extractor.parameters()):
        raise RuntimeError("Adversarial extractor not frozen/cleared for G-step")
    if any(p.requires_grad or p.grad is not None for s in objective.spaces.values() for p in s.extractor.parameters()):
        raise RuntimeError("Static extractor changed gradient scope")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite reconstruction parameters")
    if dynamic is not None:
        objective.validate_pending(dynamic)
    objective.commit(static)
    if dynamic is not None:
        objective.commit_dynamic(dynamic)
    metrics.update(loss=float(loss.detach()), raw_fd=static.raw, normalized_fd=static.normalized,
                   adv_schedule_step=step,
                   grad_norm=float(norm), group_grad_norm=norms,
                   adv_active=objective.active(step), adv_effective_weight=effective_weight,
                   critic_updated=objective.critic_due(step), critic_updates=int(objective.critic_updates),
                   clipped_pixel_fraction=float(((reconstruction.detach() <= 0) | (reconstruction.detach() >= 1)).float().mean()),
                   statistics_updates={k: int(v.statistics.updates) for k, v in objective.spaces.items()},
                   adv_statistics_updates={"real": int(objective.real_statistics.updates),
                                           "fake": int(objective.fake_statistics.updates)})
    return metrics
