"""Differentiable FD; adapted from FD-Loss (MIT), see THIRD_PARTY_NOTICES.md."""
import torch


class _FiniteSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.clamp_min(0).sqrt()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        # At the singular boundary choose zero, not infinity times zero.
        scale = torch.where(y > 1e-12, 0.5 / y.clamp_min(1e-12), 0.0)
        return grad * scale


def symmetric(x):
    return (x + x.T) * 0.5


@torch.no_grad()
def covariance_sqrt(cov):
    cov = symmetric(cov.double())
    values, vectors = torch.linalg.eigh(cov)
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T


def frechet_distance(mu_ref, cov_ref, mu, cov, ref_sqrt=None):
    """Symmetric PSD formulation. Never silently substitute a fallback loss.

    Moments/eigensolvers use float64. The reference is a fixed, detached target.
    This function is NOT the future adaptive/whitened FD implementation.
    """
    device_type = mu.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        mu, cov = mu.double(), symmetric(cov.double())
        mu_ref, cov_ref = mu_ref.detach().double(), symmetric(cov_ref.detach().double())
        if mu.shape != mu_ref.shape or cov.shape != cov_ref.shape:
            raise ValueError("Reference and current feature dimensions differ")
        if not all(torch.isfinite(t).all() for t in (mu, cov, mu_ref, cov_ref)):
            raise FloatingPointError("Non-finite FD moments")
        root = covariance_sqrt(cov_ref) if ref_sqrt is None else ref_sqrt.detach().double()
        eig = torch.linalg.eigvalsh(symmetric(root @ cov @ root))
        distance = (mu - mu_ref).square().sum() + cov.trace() + cov_ref.trace()
        distance = distance - 2 * _FiniteSqrt.apply(eig).sum()
        if not torch.isfinite(distance):
            raise FloatingPointError("Non-finite FD")
        tolerance = 1e-7 * (1 + cov.trace().abs() + cov_ref.trace().abs())
        if distance.detach() < -tolerance:
            raise FloatingPointError("Substantially negative FD; check covariance validity")
        return distance.clamp_min(0)
