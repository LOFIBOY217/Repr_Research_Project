import torch
import pytest
from recon_fd.objectives.frechet import frechet_distance, covariance_sqrt
from conftest import upstream_module


def test_matches_official_value_and_gradient(project):
    module = upstream_module(project / "third_party/FD-Loss/frechet_distance/losses.py", "official_fd")
    real = torch.randn(32, 5, dtype=torch.float64)
    generated = torch.randn(23, 5, dtype=torch.float64, requires_grad=True)
    mu_ref, cov_ref = real.mean(0), torch.cov(real.T)
    mu, cov = generated.mean(0), torch.cov(generated.T)
    root = covariance_sqrt(cov_ref)
    ours = frechet_distance(mu_ref, cov_ref, mu, cov, root)
    theirs = module.compute_frechet_distance_loss(mu_ref, cov_ref, mu=mu, sigma=cov, sigma_ref_sqrt=root)
    torch.testing.assert_close(ours.float(), theirs)
    grad_a = torch.autograd.grad(ours, generated, retain_graph=True)[0]
    grad_b = torch.autograd.grad(theirs, generated)[0]
    torch.testing.assert_close(grad_a, grad_b, atol=1e-9, rtol=1e-8)


def test_identity_and_rank_deficiency():
    x = torch.randn(4, 9, dtype=torch.float64)
    mu, cov = x.mean(0), torch.cov(x.T)
    assert frechet_distance(mu, cov, mu, cov) < 1e-6
    fake = torch.zeros(5, 9, dtype=torch.float64, requires_grad=True)
    result = frechet_distance(mu, cov, fake.mean(0), torch.cov(fake.T))
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(fake.grad).all()


def test_nonfinite_fails_not_fallback():
    with pytest.raises(FloatingPointError):
        frechet_distance(torch.zeros(3), torch.eye(3), torch.full((3,), float("nan")), torch.eye(3))
