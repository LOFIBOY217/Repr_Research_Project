import copy
import json
import pytest
from recon_fd.diagnostics.trajectory import compare_evaluations


def test_screen_is_not_proof_and_protocol_guard(tmp_path):
    a = {"checkpoint_step": 0, "dataset": {"id": "a"}, "num_samples": 50000,
         "image_protocol": "png", "representations": {"inception": "a", "dino": "b"},
         "engineering_only": False, "implementation_sha256": "test", "fd": {"inception": 10, "dino": 10}, "paired": {"psnr": 30}}
    b = copy.deepcopy(a)
    b.update(checkpoint_step=100, fd={"inception": 8, "dino": 12})
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, value in zip(paths, [a, b]):
        path.write_text(json.dumps(value))
    result = compare_evaluations(paths, "inception", ["dino"])
    assert result["comparisons"][0]["cross_representation_divergence_candidate"]
    assert "not proof" in result["warning"]
    b["num_samples"] = 10000
    paths[1].write_text(json.dumps(b))
    with pytest.raises(ValueError, match="protocols"):
        compare_evaluations(paths, "inception", ["dino"])
