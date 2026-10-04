"""AdvFD real-whitened FD, ported from 4e4cfed (MIT; see notices).

Keep the upstream diagonal loading on BOTH covariances, detached real
transform, FP64 eigensolves and FP32 return. Do not replace this with raw FD.
"""
import torch


def build_real_whitening(real_mu, real_cov, eps=1e-3):
    if eps <= 0:
        raise ValueError("Whitening epsilon must be positive")
    real_mu, real_cov = real_mu.double(), real_cov.double()
    real_cov = 0.5 * (real_cov + real_cov.T)
    eye = torch.eye(len(real_mu), device=real_mu.device, dtype=real_mu.dtype)
    eigenvalues, eigenvectors = torch.linalg.eigh(real_cov + eps * eye)
    return real_mu.detach(), eigenvectors.detach(), eigenvalues.clamp_min(eps).rsqrt().detach()


def real_whitened_frechet_distance(real_mu, real_cov, fake_mu, fake_cov, eps=1e-3):
    real_mu, vectors, inv_sqrt = build_real_whitening(real_mu, real_cov, eps)
    fake_mu, fake_cov = fake_mu.double(), fake_cov.double()
    fake_cov = 0.5 * (fake_cov + fake_cov.T)
    dim = len(fake_mu)
    mean_white = ((fake_mu - real_mu) @ vectors) * inv_sqrt
    eye = torch.eye(dim, device=fake_mu.device, dtype=fake_mu.dtype)
    cov_eigenbasis = vectors.T @ (fake_cov + eps * eye) @ vectors
    cov_white = cov_eigenbasis * inv_sqrt[:, None] * inv_sqrt[None, :]
    cov_white = 0.5 * (cov_white + cov_white.T)
    eigenvalues = torch.linalg.eigvalsh(cov_white).clamp_min(0)
    value = (mean_white.dot(mean_white) + torch.diagonal(cov_white).sum()
             + float(dim) - 2 * torch.sqrt(eigenvalues).sum()).float()
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite AdvFD whitening loss")
    return value
