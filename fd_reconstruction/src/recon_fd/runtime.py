"""Shared numerical execution controls; no changes to the FD objectives."""
import os
import torch


def configure_determinism(device):
    # Set before the first CUDA computation in every train/evaluate process.
    if torch.device(device).type == "cuda":
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def determinism_settings():
    return {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
