import pytest
import torch
from torch.nn import functional as F

from recon_fd.representations import _BicubicAntialiasResize, _resize_bicubic_antialias
from recon_fd.runtime import configure_determinism


def test_scoped_bicubic_backward_preserves_forward_gradient_and_policy():
    source = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    upstream = torch.randn(2, 3, 14, 14, dtype=torch.float64)
    expected_input = source.clone().requires_grad_()
    expected = F.interpolate(expected_input, (14, 14), mode="bicubic",
                             align_corners=False, antialias=True)
    expected_gradient, = torch.autograd.grad(expected, expected_input, upstream)

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        configure_determinism("cpu")
        actual_input = source.clone().requires_grad_()
        actual = _BicubicAntialiasResize.apply(actual_input, (14, 14))
        actual_gradient, = torch.autograd.grad(actual, actual_input, upstream)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only backward restriction")
def test_cuda_resize_backward_succeeds_under_strict_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        configure_determinism("cuda")
        images = torch.randn(2, 3, 256, 256, device="cuda", requires_grad=True)
        actual = _resize_bicubic_antialias(images, (224, 224))
        expected = F.interpolate(images.detach(), (224, 224), mode="bicubic",
                                 align_corners=False, antialias=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        assert images.grad is not None and torch.isfinite(images.grad).all()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
