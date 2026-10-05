from pathlib import Path

import pytest

from recon_fd.config import load_config, validate


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    ("filename", "method"),
    [
        ("fd_only_nibi_large.yaml", "fd_only"),
        ("advfd_nibi_large.yaml", "advfd_reconstruction"),
        ("ours_current_both_nibi_large.yaml", "ours_current_both"),
    ],
)
def test_large_configs_use_50k_and_matched_inception(filename, method):
    config = load_config(CONFIGS / filename)
    validate(config)
    assert config["method"] == method
    assert config["data"]["kind"] == "imagenet"
    assert config["train"]["batch_size"] == 16
    assert config["evaluation"]["num_samples"] == 50000
    assert config["static"]["reference_samples"] == 50000
    assert config["static"]["initialization_samples"] == 50000
    if method == "ours_current_both":
        assert not config["static"]["enabled"]
        assert config["adaptive"]["trainable_scope"] == "full"
        assert config["adaptive"]["real_stats"]["samples"] == 50000
        assert config["adaptive"]["initialization_samples"] == 50000
        assert config["adaptive"]["initialization_batch_size"] == 16
        assert config["adaptive"]["representation"]["weights"] == "${INCEPTION_WEIGHTS}"
        assert config["train"]["steps"] == 4
    else:
        assert config["static"]["enabled"]
        assert [spec["name"] for spec in config["static"]["representations"]] == ["inception"]
        assert config["static"]["representations"][0]["weights"] == "${INCEPTION_WEIGHTS}"
        assert config["train"]["steps"] == 10000
