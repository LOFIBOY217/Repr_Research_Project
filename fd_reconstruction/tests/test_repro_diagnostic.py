import importlib.util
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch


@pytest.fixture
def diagnostic(project):
    spec = importlib.util.spec_from_file_location("diagnose_repro", project / "scripts/diagnose_repro.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_difference_finds_tensor_paths_and_rng(diagnostic):
    left = {"model": torch.ones(3), "counter": torch.tensor(2), "rng": [np.array([1, 2, 3])]}
    right = {"model": torch.tensor([1., 1., 2.]), "counter": torch.tensor(3), "rng": [np.array([1, 2, 4])]}
    assert diagnostic.difference(left, left) == []
    rows = {r["path"]: r for r in diagnostic.difference(left, right)}
    assert set(rows) == {"root.model", "root.counter", "root.rng.0"}
    assert rows["root.model"]["outside_original_tolerance"] == 1
    assert rows["root.model"]["max_abs"] == 1


def test_repeated_vjp_fixture(diagnostic):
    x = torch.randn(2, 3, requires_grad=True)
    result = diagnostic.repeat_vjp(lambda: x.square(), {"x": x})
    assert result["forward_bitwise_equal"] and result["backward_bitwise_equal"]
    assert x.grad is None


def test_diagnosis_refuses_login_node(diagnostic, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="SLURM allocation"):
        diagnostic.main()


@pytest.mark.parametrize("failure", [None, "training", "strict", "missing", "error", "wrong_group"])
def test_diagnostic_exit_status_tracks_actual_controls(diagnostic, tmp_path, failure):
    probes = {mode: {name: {"forward_bitwise_equal": True, "backward_bitwise_equal": True}
                     for name in diagnostic.REQUIRED_PROBES} for mode in ("baseline", "strict")}
    # A measured baseline mismatch is expected evidence, not a diagnostic failure.
    probes["baseline"]["vae_parameters_eval"]["backward_bitwise_equal"] = False
    training = {"A": {"status": "passed"}}
    if failure == "training":
        training["A"]["status"] = "failed"
    elif failure == "strict":
        probes["strict"]["vae_parameters_eval"]["backward_bitwise_equal"] = False
    elif failure == "missing":
        del probes["strict"]["vae_parameters_eval"]
    elif failure == "error":
        probes["baseline"]["vae_parameters_eval"] = {"error": "fixture error"}
    elif failure == "wrong_group":
        training = {"B": {"status": "passed"}}
    if failure:
        with pytest.raises(SystemExit) as caught:
            diagnostic.finish_diagnosis(tmp_path, probes, training, "A")
        assert caught.value.code == 1
    else:
        diagnostic.finish_diagnosis(tmp_path, probes, training, "A")
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == ("failed" if failure else "diagnosis_completed")
    assert report["component_probes"] == probes and report["strict_training"] == training


@pytest.mark.parametrize("group,recipe", [("A", "smoke.yaml"), ("B", "advfd_smoke.yaml"),
                                         ("C", "ours_current_both_smoke.yaml")])
def test_strict_control_json_roundtrip_and_group_isolation(diagnostic, project, tmp_path, monkeypatch, group, recipe):
    from recon_fd.cli import train_main
    from recon_fd.config import load_config
    from recon_fd.provenance import write_json
    monkeypatch.syspath_prepend(str(project / "scripts"))
    source = tmp_path / "source"
    monkeypatch.setenv("FD_AB_SOURCE", str(source))
    monkeypatch.setenv("FD_C_SOURCE", str(source))
    config = load_config(project / "configs" / recipe)
    config["train"]["lr"] = 1e-6
    config["static"]["reference_cache"] = str(tmp_path / "cache")
    parent = source if group == "C" else source / group
    write_json(parent / "real_resumed/config.json", config)
    calls = []

    def local_worker(command, **kwargs):
        # CPU integration of the same CLI; do not pretend to hold a SLURM GPU.
        calls.append(command)
        train_main(command[4:])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(diagnostic.subprocess, "run", local_worker)
    result = diagnostic.strict_training(tmp_path, group)
    assert set(result) == {group} and result[group]["status"] == "passed"
    assert result[group]["stages"] == ["train", "resume", "continuous"] and len(calls) == 3
    assert all(v["bitwise_equal"] for v in result[group]["comparisons"].values())
    assert {p.name for p in tmp_path.glob("strict_*") if p.is_dir()} == {f"strict_{group}"}
