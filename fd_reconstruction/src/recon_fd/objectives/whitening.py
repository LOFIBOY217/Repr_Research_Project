"""Call the unchanged official AdvFD whitening kernel, with failure checks."""
import torch
from recon_fd.vendor.advfd.adversarial import (
    build_real_whitening as _build_real_whitening,
    real_whitened_frechet_distance_from_stats,
)


def build_real_whitening(real_mu, real_cov, eps=1e-3):
    if eps <= 0:
        raise ValueError("Whitening epsilon must be positive")
    return _build_real_whitening(real_mu, real_cov, eps=eps)


def real_whitened_frechet_distance(real_mu, real_cov, fake_mu, fake_cov, eps=1e-3):
    if eps <= 0:
        raise ValueError("Whitening epsilon must be positive")
    value = real_whitened_frechet_distance_from_stats(real_mu, real_cov, fake_mu, fake_cov, eps=eps)
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite AdvFD whitening loss")
    return value
