import json
import pytest
import torch
from recon_fd.cli import train_main, evaluate_main
from recon_fd.engine.checkpoint import read_checkpoint


def test_end_to_end_train_resume_evaluate(project, tmp_path):
    config = str(project / "configs/smoke.yaml")
    cache = str(tmp_path / "cache")
    common = ["--config", config, "--set", f"static.reference_cache={cache}"]
    run = tmp_path / "training"
    train_main([*common, "--set", f"train.output={run}", "--set", "train.steps=2"])
    saved = run / "checkpoints/step_0000002.pt"
    train_main([*common, "--set", f"train.output={run}", "--set", "train.steps=4", "--resume", str(saved)])
    final = run / "checkpoints/step_0000004.pt"
    assert read_checkpoint(final)["step"] == 4
    continuous = tmp_path / "continuous"
    train_main([*common, "--set", f"train.output={continuous}", "--set", "train.steps=4"])
    a = read_checkpoint(final)
    b = read_checkpoint(continuous / "checkpoints/step_0000004.pt")
    for component in ("model", "objective"):
        for key in a[component]:
            torch.testing.assert_close(a[component][key], b[component][key], rtol=0, atol=0)
    evaluation = tmp_path / "evaluation"
    evaluate_main([*common, "--checkpoint", str(final), "--output", str(evaluation)])
    metrics = json.loads((evaluation / "metrics.json").read_text())
    assert metrics["num_samples"] == 32 and metrics["engineering_only"] is True
    assert set(metrics["fd"]) == {"tiny", "tiny_holdout"}
    assert metrics["checkpoint_step"] == 4
    assert len((evaluation / "per_image.csv").read_text().splitlines()) == 33


@pytest.mark.parametrize("mode", ["queue", "queue_online"])
def test_queue_cli(project, tmp_path, mode):
    train_main(["--config", str(project / "configs/smoke.yaml"),
                "--set", f"train.output={tmp_path / 'queue'}", "--set", f"static.statistics={mode}",
                "--set", f"static.reference_cache={tmp_path / 'cache'}", "--set", "train.steps=2"])
    state = read_checkpoint(tmp_path / "queue/checkpoints/step_0000002.pt")
    assert int(state["objective"]["spaces.tiny.statistics.updates"]) == 2
