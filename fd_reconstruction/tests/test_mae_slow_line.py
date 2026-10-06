"""The slow-critic MAE line is separate from official B and exact-pool C."""
import json
import math
from pathlib import Path

import pytest
import torch

from recon_fd.cli import train_main
from recon_fd.config import load_config, validate
from recon_fd.data import SyntheticDataset
from recon_fd.engine.trainer import build_objective, initialize_statistics
from recon_fd.objectives.current_batch import CurrentBatchFD
from recon_fd.objectives.whitening import real_whitened_frechet_distance
from recon_fd.tokenizers import TinyReconstructor


def test_mae_b_c_share_training_protocol_except_the_declared_method_changes(project):
    b = load_config(project / "configs/mae_slow_b.yaml")
    c = load_config(project / "configs/mae_slow_c.yaml")
    validate(b)
    validate(c)
    assert b["data"] == c["data"] and b["tokenizer"] == c["tokenizer"]
    assert {k: v for k, v in b["train"].items() if k != "output"} == {
        k: v for k, v in c["train"].items() if k != "output"}
    for key in ("weight", "ema_beta", "whiten_eps", "lr", "betas", "weight_decay", "grad_clip",
                "start_step", "warmup_steps", "first_critic_step", "update_freq", "steps_per_update",
                "gradient_checkpointing", "lora"):
        assert b["adaptive"][key] == c["adaptive"][key], key
    assert b["adaptive"]["first_critic_step"] == 50000 // b["train"]["batch_size"] == 3125
    assert b["adaptive"]["representation"] == "mae"
    assert next(s for s in b["static"]["representations"] if s["name"] == "mae")["model_name"] == (
        c["adaptive"]["representation"]["model_name"])
    assert b["adaptive"]["trainable_scope"] == "paper"
    assert c["adaptive"]["trainable_scope"] == "full"
    assert b["static"]["enabled"] and not c["static"]["enabled"]
    assert b["adaptive"]["real_stats"]["mode"] == "ema"
    assert c["adaptive"]["real_stats"]["mode"] == "reencode_pool"
    assert c["adaptive"]["fake_stats"] == {"mode": "batch_current"}


def smoke_system(project):
    config = load_config(project / "configs/mae_slow_c_smoke.yaml")
    validate(config)
    dataset = SyntheticDataset(64, 16, 217)
    model = TinyReconstructor()
    objective, _ = build_objective(config, dataset, "cpu")
    initialize_statistics(model, objective, dataset, config, "cpu")
    images = torch.stack([dataset[i]["image"] for i in range(config["train"]["batch_size"])])
    return config, model, objective, images


def test_current_batch_has_no_fake_ema_and_exact_current_minibatch_moments(project):
    config, model, objective, images = smoke_system(project)
    assert isinstance(objective, CurrentBatchFD)
    assert objective.fake_statistics is None and objective.real_reference.ema is None
    assert not any("ema" in key for key in objective.state_dict())
    assert objective.parameter_manifest()["fake_statistics"] == "current_minibatch_no_ema"
    reconstruction = model(images)
    actual = objective.dynamic(images, reconstruction, 0)
    fake_features = objective.extractor(reconstruction)
    real = objective.real_reference.preview(objective.extractor, None, 0).moments
    expected = real_whitened_frechet_distance(real.mean, real.cov, fake_features.mean(0),
                                              torch.cov(fake_features.T), config["adaptive"]["whiten_eps"])
    torch.testing.assert_close(actual.fd, expected)
    assert actual.fake_features.shape[0] == len(images)
    with torch.no_grad():
        model.decoder[0].bias.add_(0.05)
    changed = objective.dynamic(images, model(images), 0)
    assert not torch.equal(actual.fd, changed.fd)
    assert int(objective.real_reference.refreshes) == 1
    assert objective.critic_due(0) is False
    assert objective.critic_due(1) is False
    assert objective.critic_due(2) is True
    assert objective.critic_due(3) is False
    assert objective.critic_due(4) is True


def test_slow_critic_cpu_train_records_current_real_and_fake_scope(project, tmp_path):
    output = tmp_path / "slow_c"
    train_main(["--config", str(project / "configs/mae_slow_c_smoke.yaml"),
                "--set", f"train.output={output}"])
    rows = [json.loads(s) for s in (output / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert [row["critic_updated"] for row in rows] == [False, False, True, False, True]
    assert [row["critic_updates"] for row in rows] == [0, 0, 1, 1, 2]
    assert [row["real_reference_refreshes"] for row in rows] == [1, 1, 2, 2, 3]
    for row in rows:
        assert row["fd_batch_size"] == 8
        assert row["fake_statistics_samples"] == 8
        assert row["fake_ema_updates"] == row["real_ema_updates"] == 0
        assert row["fake_statistics_historical"] is False
        assert row["real_feature_version"] == row["loss_psi_version"]
        assert math.isfinite(row["adv_fd"]) and math.isfinite(row["grad_norm"])
    assert (output / "checkpoints/step_0000005.pt").is_file()


def test_slow_critic_b_stays_frozen_until_the_declared_boundary(project, tmp_path):
    output = tmp_path / "slow_b"
    train_main(["--config", str(project / "configs/mae_slow_b_smoke.yaml"),
                "--set", f"train.output={output}"])
    rows = [json.loads(s) for s in (output / "train.jsonl").read_text().splitlines()]
    assert [row["critic_updated"] for row in rows] == [False, False, True, False, True]
    assert [row["critic_updates"] for row in rows] == [0, 0, 1, 1, 2]
    assert all(row["adv_active"] for row in rows)


def test_slow_critic_c_resumes_without_losing_psi_or_reference_versions(project, tmp_path):
    config = str(project / "configs/mae_slow_c_smoke.yaml")
    output = tmp_path / "resume_c"
    train_main(["--config", config, "--set", f"train.output={output}",
                "--set", "train.steps=2"])
    checkpoint = output / "checkpoints/step_0000002.pt"
    train_main(["--config", config, "--set", f"train.output={output}",
                "--resume", str(checkpoint)])
    rows = [json.loads(s) for s in (output / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["critic_updates"] for row in rows] == [0, 0, 1, 1, 2]
    assert [row["real_feature_version"] for row in rows] == [0, 0, 1, 1, 2]


def test_slow_critic_invalid_modes_rejected(project):
    smoke = project / "configs/mae_slow_c_smoke.yaml"
    for override in ("adaptive.fake_stats.mode=reencode_pool", "adaptive.first_critic_step=-1",
                     "adaptive.initialization_batch_size=7"):
        with pytest.raises(ValueError):
            validate(load_config(smoke, [override]))
