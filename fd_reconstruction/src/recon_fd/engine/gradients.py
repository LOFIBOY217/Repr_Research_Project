"""Upstream G gradient policy; reject non-finite gradients before any step."""
import torch
from recon_fd.vendor.fd_loss.grad_util import get_grad_norm


def checked_grad_norm(parameters, clip):
    parameters = list(parameters)
    if clip < 0:
        raise ValueError("Gradient clipping must be nonnegative; zero disables it")
    norm = (torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
            if clip > 0 else get_grad_norm(parameters))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite generator gradient norm")
    return norm
