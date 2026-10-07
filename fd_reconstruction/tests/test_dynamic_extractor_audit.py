"""Checkpoint audit tests use tiny synthetic models, never MAE science claims."""
import importlib.util
from pathlib import Path

import pytest
import torch

from recon_fd.cli import train_main
from recon_fd.engine.checkpoint import read_checkpoint
from recon_fd.representations import build_representation
from recon_fd.representations.lora import apply_qkv_lora


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_dynamic_extractor.py"
SPEC = importlib.util.spec_from_file_location("audit_dynamic_extractor", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def train_preflight(project, tmp_path, group):
    run = tmp_path / group
    if group == "B":
        config = project / "configs/mae_slow_b_smoke.yaml"
        overrides = ["adaptive.first_critic_step=1", f"static.reference_cache={tmp_path / 'static'}"]
    else:
        config = project / "configs/ours_current_both_smoke.yaml"
        overrides = []
    train_main(["--config", str(config), "--set", f"train.output={run}",
                "--set", "train.steps=2", "--set", "train.save_every=1",
                *[value for override in overrides for value in ("--set", override)]])
    return run


@pytest.mark.parametrize("group,updates", [("B", [0, 1]), ("C", [1, 0])])
def test_audit_detects_actual_D_only_extractor_updates(project, tmp_path, group, updates):
    run = train_preflight(project, tmp_path, group)
    result = audit.audit_run(run, device="cpu", probe_count=4, require_mae=False)
    assert result["status"] == "passed"
    assert [row["critic_updates_in_step"] for row in result["checks"]] == updates
    for row in result["checks"]:
        assert not row["unexpected_frozen_changes"]
        assert (row["changed_trainable_tensors"] > 0) == bool(row["critic_updates_in_step"])


def test_audit_rejects_false_update_counter(project, tmp_path):
    run = train_preflight(project, tmp_path, "C")
    first = read_checkpoint(run / "checkpoints/step_0000000.pt")
    second_path = run / "checkpoints/step_0000001.pt"
    second = read_checkpoint(second_path)
    for key in audit.extractor_state(second):
        second["objective"]["extractor." + key] = first["objective"]["extractor." + key]
    torch.save(second, second_path)
    with pytest.raises(AssertionError, match="no trainable MAE parameter changed"):
        audit.audit_run(run, device="cpu", probe_count=4, require_mae=False)


def test_audit_rebuilds_lora_extractor_from_saved_dynamic_state(tmp_path):
    timm = pytest.importorskip("timm")
    raw = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    weights = tmp_path / "vit.pt"
    torch.save(raw.state_dict(), weights)
    spec = {"name": "vit", "kind": "timm", "model_name": "vit_tiny_patch16_224",
            "weights": str(weights), "target_size": 32, "pool": "cls"}
    original = build_representation(spec)
    apply_qkv_lora(original, rank=16, alpha=16.0, dropout=0.0)
    with torch.no_grad():
        next(parameter for name, parameter in original.named_parameters() if ".lora_B." in name).add_(0.01)
    config = {"method": "advfd_reconstruction", "static": {"representations": [spec]},
              "adaptive": {"representation": "vit", "lora": {"rank": 16, "alpha": 16.0, "dropout": 0.0}}}
    rebuilt = audit.build_dynamic_extractor(config, original.state_dict(), "cpu")
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(images), original.eval()(images), rtol=0, atol=0)
