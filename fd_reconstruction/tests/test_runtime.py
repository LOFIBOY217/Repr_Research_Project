import pytest
import torch
from recon_fd.runtime import configure_determinism, determinism_settings


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_runtime_enforces_strict_controls_without_cuda_compute(monkeypatch, device):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    configure_determinism(device)
    settings = determinism_settings()
    assert settings["algorithms"] and not settings["warn_only"]
    assert settings["cudnn_deterministic"] and not settings["cudnn_benchmark"]
    assert not settings["cudnn_allow_tf32"] and not settings["matmul_allow_tf32"]
    assert settings["cublas_workspace_config"] == (":4096:8" if device == "cuda" else None)


def test_invalid_workspace_configuration_rejected(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        configure_determinism("cuda")


def test_supported_existing_workspace_is_preserved(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    configure_determinism("cuda")
    assert determinism_settings()["cublas_workspace_config"] == ":16:8"
