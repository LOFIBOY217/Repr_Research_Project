import os
import random
from pathlib import Path
import numpy as np
import torch
from recon_fd.provenance import implementation_fingerprint


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Cannot exactly resume a CUDA RNG state on CPU")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, model, objective, optimizer, step, signature, data_identity, config, critic_optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    state = {"schema": 1, "step": step, "model": model.state_dict(), "objective": objective.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": rng_state(), "signature": signature,
                "data_identity": data_identity, "config": config,
                "implementation_sha256": implementation_fingerprint()}
    if critic_optimizer is not None:
        state["critic_optimizer"] = critic_optimizer.state_dict()
    torch.save(state, temporary)
    os.replace(temporary, path)


def read_checkpoint(path):
    # Training checkpoints include Python/NumPy RNG state. Load only our own
    # trusted local artifacts, never arbitrary downloaded pickle checkpoints.
    result = torch.load(path, map_location="cpu", weights_only=False)
    if result.get("schema") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return result


def restore_checkpoint(state, model, objective, optimizer, signature, data_identity, critic_optimizer=None):
    if state["signature"] != signature or state["data_identity"] != data_identity:
        raise ValueError("Resume configuration or dataset changed")
    if state.get("implementation_sha256") != implementation_fingerprint():
        raise ValueError("Implementation changed since checkpoint; exact resume is not safe")
    if (critic_optimizer is not None) != ("critic_optimizer" in state):
        raise ValueError("Checkpoint adversarial optimizer missing or unexpected")
    model.load_state_dict(state["model"], strict=True)
    objective.load_state_dict(state["objective"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if critic_optimizer is not None:
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    restore_rng(state["rng"])
    return int(state["step"])
