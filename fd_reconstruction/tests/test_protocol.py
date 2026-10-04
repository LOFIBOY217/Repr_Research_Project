import pytest
import torch
from recon_fd.config import load_config, validate
from recon_fd.data import SyntheticDataset, selected_dataset, ResumableBatchSampler
from recon_fd.evaluation.runner import export_reconstructions, verify_manifest, psnr_per_image
from recon_fd.evaluation.reference import get_reference, reference_path, load_reference
from recon_fd.objectives.frechet import frechet_distance
from recon_fd.representations import build_representation
from recon_fd.tokenizers import TinyReconstructor


def test_config_modes_and_50k_guard(project):
    config = load_config(project / "configs/smoke.yaml")
    validate(config)
    config["evaluation"]["engineering_only"] = False
    with pytest.raises(ValueError, match="50,000"):
        validate(config)
    config = load_config(project / "configs/smoke.yaml", ["method=ours", "adaptive.enabled=true"])
    with pytest.raises(ValueError):  # A config cannot silently become C by changing only its name.
        validate(config)
    with pytest.raises(ValueError, match="Unknown override"):
        load_config(project / "configs/smoke.yaml", ["train.typo=5"])


def test_resume_sample_order():
    whole = list(ResumableBatchSampler(35, 8, 99, 0, 14))
    tail = list(ResumableBatchSampler(35, 8, 99, 5, 14))
    assert whole[5:] == tail


def test_no_silent_sample_truncation():
    dataset = SyntheticDataset(20, 8, 4)
    with pytest.raises(ValueError, match="no silent truncation"):
        selected_dataset(dataset, 50000, 0)


def test_export_and_missing_image_rejected(project, tmp_path):
    config = load_config(project / "configs/smoke.yaml")
    dataset = selected_dataset(SyntheticDataset(10, 8, 123), 10, 0)
    model = TinyReconstructor()
    result = export_reconstructions(model, dataset, tmp_path, config, torch.device("cpu"))
    assert result["count"] == 10
    verify_manifest(tmp_path, dataset.ids, result["identity"])
    (tmp_path / "000004.png").unlink()
    with pytest.raises(ValueError, match="Missing"):
        verify_manifest(tmp_path, dataset.ids, result["identity"])


def test_reference_provenance_rejects_mismatch(tmp_path):
    dataset = selected_dataset(SyntheticDataset(10, 8, 123), 10, 0)
    representation = build_representation({"name": "tiny", "kind": "tiny", "seed": 0})
    moments, identity = get_reference(representation, dataset, tmp_path, 4, "cpu")
    path = reference_path(tmp_path, identity)
    load_reference(path, identity, "cpu")
    changed = {**identity, "pixel_protocol": "wrong"}
    with pytest.raises(ValueError, match="mismatch"):
        load_reference(path, changed, "cpu")


def test_pair_permutation_invisible_to_fd_not_pair_metrics():
    real = torch.rand(20, 3, 8, 8)
    shuffled = real.roll(1, 0)
    representation = build_representation({"kind": "tiny", "name": "tiny"})
    a, b = representation(real).double(), representation(shuffled).double()
    assert frechet_distance(a.mean(0), torch.cov(a.T), b.mean(0), torch.cov(b.T)) < 1e-8
    assert psnr_per_image(real, real).mean() > psnr_per_image(real, shuffled).mean() + 50
