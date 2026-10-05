import importlib.util
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
