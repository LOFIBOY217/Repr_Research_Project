"""Exercise the exact A/B acceptance protocol locally, without a SLURM fiction."""
from copy import deepcopy
import importlib.util
import json
import pytest
import torch
from recon_fd.cli import train_main, evaluate_main
from recon_fd.config import load_config, validate
from recon_fd.engine import trainer
from recon_fd.engine.checkpoint import read_checkpoint


@pytest.fixture
def acceptance(project):
    spec = importlib.util.spec_from_file_location("ab_acceptance", project / "scripts/ab_acceptance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_configs_change_only_explicit_smoke_protocol(project):
    a = load_config(project / "configs/fd_only_acceptance_nibi.yaml")
    b = load_config(project / "configs/advfd_acceptance_nibi.yaml")
    official = load_config(project / "configs/advfd_reconstruction_inception.yaml")
    for config in (a, b):
        validate(config)
        assert config["evaluation"]["engineering_only"]
        assert config["static"]["reference_samples"] == config["static"]["queue_size"] == 128
        assert config["static"]["initialization_samples"] == 128
        assert config["evaluation"]["num_samples"] == 32
        assert config["train"]["batch_size"] == 2 and config["train"]["grad_clip"] == 0
    assert a["static"] == b["static"]
    assert a["data"] == b["data"] and a["tokenizer"] == b["tokenizer"]
    assert a["evaluation"] == b["evaluation"]
    for key in ("lr", "betas", "weight_decay"):
        assert a["train"][key] == b["train"][key]
    expected = {**official["adaptive"], "start_step": 2, "warmup_steps": 2}
    assert b["adaptive"] == expected
    assert official["adaptive"]["start_step"] == 1000 and official["adaptive"]["warmup_steps"] == 4000


@pytest.mark.parametrize("group,filename", [("A", "smoke.yaml"), ("B", "advfd_smoke.yaml")])
def test_acceptance_actual_protocol_and_verifier(acceptance, project, tmp_path, monkeypatch, group, filename):
    records = []
    original = trainer.adversarial_g_step
    monkeypatch.setattr(trainer, "adversarial_g_step", acceptance.audited_b_step(original, records))
    common = ["--config", str(project / "configs" / filename), "--set", "runtime.device=cpu",
              "--set", "train.save_every=2", "--set", "train.log_every=1",
              "--set", f"static.reference_cache={tmp_path / 'cache'}"]

    def run(stage, arguments):
        records.clear()
        train_main(arguments)
        (tmp_path / f"{stage}.trace.json").write_text(json.dumps(records))

    comparisons = acceptance.protocol(tmp_path, "tiny", common, run)
    assert set(comparisons) == {"0", "2", "4", "6"}
    assert all(v["passed"] and v["bitwise_equal"] for v in comparisons.values())
    checks = acceptance.verify_run(tmp_path, group, "tiny")
    assert checks["B_D_then_G_and_gradient_isolation_verified"] == (group == "B")
    evaluations = []
    for step in (0, 6):
        output = tmp_path / f"evaluation_step{step}"
        evaluate_main([*common, "--checkpoint",
            str(tmp_path / f"tiny_resumed/checkpoints/step_{step:07d}.pt"), "--output", str(output)])
        evaluations.append(acceptance.verify_evaluation(output, step, ["tiny", "tiny_holdout"]))
    assert evaluations[0]["sample_ids"] == evaluations[1]["sample_ids"]
    assert evaluations[0]["representations"] == evaluations[1]["representations"]

    checkpoint = read_checkpoint(tmp_path / "tiny_resumed/checkpoints/step_0000006.pt")
    changed = deepcopy(checkpoint)
    key = next(iter(changed["model"]))
    changed["model"][key].add_(1)
    with pytest.raises(AssertionError):
        acceptance.compare_checkpoints(checkpoint, changed)
    changed = deepcopy(checkpoint)
    changed["rng"]["torch"][0] ^= 1
    with pytest.raises(AssertionError):
        acceptance.compare_checkpoints(checkpoint, changed)
    changed = deepcopy(checkpoint)
    if group == "B":
        del changed["critic_optimizer"]
    else:
        changed["critic_optimizer"] = {}
    with pytest.raises(AssertionError):
        acceptance.compare_checkpoints(checkpoint, changed)


def test_acceptance_requires_allocation_before_ml(acceptance, monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    for entry in (acceptance.main, acceptance.worker, lambda: acceptance.group_main("A", tmp_path)):
        with pytest.raises(RuntimeError, match="no ML on login nodes"):
            entry()


def test_audit_restores_hooks_when_training_fails(acceptance, official_small_system, project):
    from recon_fd.objectives.adaptive_fd import AdvFD
    model, static, g_opt, images = official_small_system
    objective = AdvFD(static, load_config(project / "configs/advfd_smoke.yaml")["adaptive"])
    d_opt = torch.optim.AdamW(objective.critic_parameters(), lr=1e-6)
    bindings = [(owner, name, name in owner.__dict__, owner.__dict__.get(name))
                for owner, name in ((model, "forward"), (g_opt, "step"), (d_opt, "step"),
                                    (objective, "commit"), (objective, "commit_dynamic"))]
    records = []
    def fails(*args):
        raise ValueError("Expected fixture failure")
    with pytest.raises(ValueError, match="Expected fixture failure"):
        acceptance.audited_b_step(fails, records)(model, objective, g_opt, d_opt, images, 0, 0)
    assert not records
    for owner, name, existed, value in bindings:
        assert (name in owner.__dict__) == existed and owner.__dict__.get(name) is value
