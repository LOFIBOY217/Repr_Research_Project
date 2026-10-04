"""Scheduled, single-GPU engineering smoke; never run on a login node."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def prepare_images(source, destination):
    """Small deterministic class-diverse subset, referencing unchanged images."""
    records = []
    for split, count in (("train", 128), ("val", 32)):
        root = source / split
        classes = sorted(p for p in root.iterdir() if p.is_dir())
        rng = random.Random(217 + (split == "val"))
        if len(classes) < count:
            raise ValueError(f"Expected class-organized ImageNet with >= {count} classes: {root}")
        for directory in rng.sample(classes, count):
            files = sorted(p for p in directory.iterdir()
                           if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg", ".png"})
            if not files:
                raise ValueError(f"Empty ImageNet class: {directory}")
            original = rng.choice(files)
            target = destination / split / directory.name / original.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(original.resolve())
            records.append({"split": split, "sample_id": f"{directory.name}/{original.name}",
                            "source": str(original.resolve())})
    return records


def worker():
    import torch
    from recon_fd.cli import train_main, evaluate_main
    from recon_fd.provenance import write_json
    stage, root, *arguments = sys.argv[2:]
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    succeeded = False
    try:
        (evaluate_main if stage.startswith("evaluate") else train_main)(arguments)
        succeeded = True
    finally:
        write_json(Path(root) / f"{stage}.resources.json", {
            "succeeded": succeeded, "seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        })


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use jobs/smoke_nibi.sbatch; this must run in a SLURM allocation")
    import torch
    from recon_fd.engine.checkpoint import read_checkpoint
    from recon_fd.provenance import implementation_fingerprint, write_json

    root = Path(os.environ["FD_SMOKE_ROOT"])
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages = []
    active_stage = "preflight"
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Smoke requires exactly one visible CUDA GPU")
        # Fail before training if a local pretrained resource is incomplete.
        tokenizer = Path(os.environ["TOKENIZER_CHECKPOINT"])
        for path in (tokenizer / "config.json", tokenizer / "diffusion_pytorch_model.safetensors",
                     Path(os.environ["INCEPTION_WEIGHTS"])):
            if not path.is_file():
                raise FileNotFoundError(path)
        write_json(root / "environment.json", {
            "job_id": os.environ["SLURM_JOB_ID"], "hostname": os.uname().nodename,
            "python": sys.version, "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda, "implementation_sha256": implementation_fingerprint(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torchvision", "timm", "diffusers", "numpy", "Pillow", "scikit-image")},
            "engineering_only": True,
        })
        records = prepare_images(Path(os.environ["IMAGENET_ROOT"]), root / "images")
        write_json(root / "image_manifest.json", {"engineering_only": True, "samples": records})

        def run(stage, arguments):
            nonlocal active_stage
            active_stage = stage
            print(f"START {stage}", flush=True)
            log = root / f"{stage}.log"
            with log.open("w") as handle:
                result = subprocess.run([sys.executable, "-u", __file__, "worker", stage, str(root), *arguments],
                                        stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-10000:], flush=True)
                raise RuntimeError(f"{stage} failed (exit {result.returncode}); inspect {log}")
            stages.append(stage)
            print(f"PASS {stage}: {log}", flush=True)

        tiny = ["--config", "configs/smoke.yaml", "--set", "runtime.device=cuda",
                "--set", f"static.reference_cache={root / 'tiny_cache'}",
                "--set", f"train.output={root / 'tiny_training'}"]
        run("tiny_train", [*tiny, "--set", "train.steps=2"])
        run("tiny_resume", [*tiny, "--set", "train.steps=4", "--resume",
                            str(root / "tiny_training/checkpoints/step_0000002.pt")])
        run("evaluate_tiny", [*tiny, "--checkpoint", str(root / "tiny_training/checkpoints/step_0000004.pt"),
                              "--output", str(root / "tiny_evaluation")])
        real = ["--config", "configs/nibi_smoke.yaml"]
        run("real_train", real)
        run("real_resume", [*real, "--set", "train.steps=6", "--resume",
                            str(root / "real_training/checkpoints/step_0000003.pt")])
        for step in (0, 6):
            run(f"evaluate_real_{step}", [*real, "--checkpoint",
                str(root / f"real_training/checkpoints/step_{step:07d}.pt"),
                "--output", str(root / f"evaluation_step{step}")])

        active_stage = "verify"
        rows = [json.loads(line) for line in (root / "real_training/train.jsonl").read_text().splitlines()]
        if [row["step"] for row in rows] != list(range(1, 7)):
            raise AssertionError("Resume lost or duplicated training steps")
        for row in rows:
            assert row["statistics_updates"] == {"inception": row["step"]}
            assert all(math.isfinite(v) and v > 0 for v in row["group_grad_norm"].values())
            assert math.isfinite(row["raw_fd"]["inception"])
        initial = read_checkpoint(root / "real_training/checkpoints/step_0000000.pt")
        final = read_checkpoint(root / "real_training/checkpoints/step_0000006.pt")
        changed = [key for key in initial["model"]
                   if not torch.equal(initial["model"][key], final["model"][key])]
        assert any("encoder." in key for key in changed), "Encoder did not update"
        assert any("decoder." in key for key in changed), "Decoder did not update"
        for key in initial["objective"]:
            if ".extractor." in key or ".reference_" in key:
                assert torch.equal(initial["objective"][key], final["objective"][key]), key
        evaluations = {}
        for step in (0, 6):
            folder = root / f"evaluation_step{step}"
            metrics = json.loads((folder / "metrics.json").read_text())
            assert metrics["engineering_only"] and metrics["num_samples"] == 32
            assert metrics["checkpoint_step"] == step
            assert set(metrics["fd"]) == {"inception", "dinov2"}
            assert all(math.isfinite(value) for value in metrics["fd"].values())
            assert len(list((folder / "reconstructions").glob("*.png"))) == 32
            assert len((folder / "per_image.csv").read_text().splitlines()) == 33
            evaluations[str(step)] = {"fd": metrics["fd"], "paired": metrics["paired"]}
        write_json(root / "result.json", {
            "status": "passed", "engineering_only": True, "job_id": os.environ["SLURM_JOB_ID"],
            "stages": stages, "seconds": time.monotonic() - started, "steps": 6,
            "train_reference_images": 128, "evaluation_images_per_checkpoint": 32,
            "encoder_and_decoder_updated": True, "frozen_extractor_unchanged": True,
            "reference_unchanged": True, "evaluations": evaluations,
            "boundary": "Engineering smoke only; not 50k evaluation or evidence about hacking.",
        })
        print(f"ALL CHECKS PASSED: {root / 'result.json'}", flush=True)
    except Exception as error:
        write_json(root / "result.json", {"status": "failed", "engineering_only": True,
                   "stage": active_stage, "stages_passed": stages, "error": repr(error)})
        raise


if __name__ == "__main__":
    worker() if len(sys.argv) > 1 and sys.argv[1] == "worker" else main()
