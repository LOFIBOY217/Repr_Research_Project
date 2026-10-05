"""One independently selected A or B acceptance per SLURM job/GPU.

This file audits the existing training path without modifying its algorithms.
No default combined A/B execution. C has its own current-both controller.
"""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def compare_checkpoints(left, right):
    """State, optimizer, RNG and metadata; tolerate only floating GPU rounding."""
    import numpy as np
    import torch
    bitwise = True

    def compare(a, b):
        nonlocal bitwise
        if isinstance(a, torch.Tensor):
            bitwise = bitwise and torch.equal(a, b)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert type(a) is type(b) and len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b

    fields = ["model", "objective", "optimizer", "rng", "signature", "data_identity",
              "implementation_sha256", "schema", "step"]
    assert ("critic_optimizer" in left) == ("critic_optimizer" in right)
    if "critic_optimizer" in left:
        fields.append("critic_optimizer")
    for field in fields:
        compare(left[field], right[field])
    return {"passed": True, "bitwise_equal": bitwise, "rtol": 1e-5, "atol": 1e-7,
            "fields": fields}


def audited_b_step(original, records):
    """Observe the real D/G optimizer calls, forwards, commits and grad scopes."""
    def wrapped(model, objective, optimizer, critic_optimizer, images, grad_clip, step):
        events, restored = [], []

        def instrument(owner, name, label, check=None):
            function = getattr(owner, name)
            def call(*args, **kwargs):
                if check is not None:
                    check()
                events.append(label)
                return function(*args, **kwargs)
            # Restore the exact attribute binding, including descriptor lookup.
            restored.append((owner, name, name in owner.__dict__, owner.__dict__.get(name)))
            setattr(owner, name, call)

        def check_d():
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert all(not p.requires_grad and p.grad is None for p in objective.static.parameters())
            assert any(p.grad is not None for p in objective.critic_parameters())

        def check_g():
            assert all(not p.requires_grad and p.grad is None for p in objective.extractor.parameters())
            assert all(not p.requires_grad and p.grad is None for p in objective.static.parameters())
            for parameters in model.parameter_groups().values():
                assert any(p.requires_grad and p.grad is not None for p in parameters)

        instrument(model, "forward", "reconstruct")
        instrument(critic_optimizer, "step", "D", check_d)
        instrument(optimizer, "step", "G", check_g)
        instrument(objective, "commit", "static_commit")
        instrument(objective, "commit_dynamic", "dynamic_commit")
        try:
            result = original(model, objective, optimizer, critic_optimizer, images, grad_clip, step)
            expected = ["reconstruct"]
            if objective.critic_due(step):
                expected += ["D"] * objective.config["steps_per_update"]
            expected += ["G", "static_commit"]
            if objective.active(step):
                expected += ["dynamic_commit"]
            assert events == expected, (step, events, expected)
            records.append({"schedule_step": step, "events": events, "gradient_scopes_checked": True})
            return result
        finally:
            for owner, name, existed, value in reversed(restored):
                if existed:
                    setattr(owner, name, value)
                else:
                    delattr(owner, name)
    return wrapped


def protocol(root, label, common, run):
    """Resume before AND after B activation; compare against independent training."""
    from recon_fd.engine.checkpoint import read_checkpoint
    resumed, continuous = root / f"{label}_resumed", root / f"{label}_continuous"
    for start, stop in ((0, 2), (2, 4), (4, 6)):
        arguments = [*common, "--set", f"train.output={resumed}", "--set", f"train.steps={stop}"]
        if start:
            # Exercise the actual saved JSON, not only the original YAML recipe.
            arguments += ["--config", str(resumed / "config.json"),
                          "--resume", str(resumed / f"checkpoints/step_{start:07d}.pt")]
        run(f"{label}_to_{stop}", arguments)
    run(f"{label}_continuous", [*common, "--set", f"train.output={continuous}", "--set", "train.steps=6"])
    comparisons = {}
    for step in (0, 2, 4, 6):
        comparisons[str(step)] = compare_checkpoints(
            read_checkpoint(resumed / f"checkpoints/step_{step:07d}.pt"),
            read_checkpoint(continuous / f"checkpoints/step_{step:07d}.pt"))
    left = [json.loads(s) for s in (resumed / "train.jsonl").read_text().splitlines()]
    right = [json.loads(s) for s in (continuous / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in left] == [r["step"] for r in right] == list(range(1, 7))
    assert [r["sample_ids"] for r in left] == [r["sample_ids"] for r in right]
    return comparisons


def verify_run(root, group, label):
    import torch
    from recon_fd.engine.checkpoint import read_checkpoint
    run = root / f"{label}_resumed"
    first = read_checkpoint(run / "checkpoints/step_0000000.pt")
    final = read_checkpoint(run / "checkpoints/step_0000006.pt")
    rows = [json.loads(s) for s in (run / "train.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, 7))
    names = {s["name"] for s in final["config"]["static"]["representations"]}
    for row in rows:
        assert set(row["raw_fd"]) == names
        assert row["statistics_updates"] == dict.fromkeys(names, row["step"])
        assert math.isfinite(row["loss"]) and all(math.isfinite(v) for v in row["raw_fd"].values())
        assert all(math.isfinite(v) and v > 0 for v in row["group_grad_norm"].values())
        assert row["fd_batch_size"] == final["config"]["train"]["batch_size"]
    for part in ("encoder", "decoder"):
        assert any(not torch.equal(first["model"][n], v) for n, v in final["model"].items()
                   if f"{part}." in n), part
    prefix = "static.spaces." if group == "B" else "spaces."
    frozen = [n for n in first["objective"] if n.startswith(prefix)
              and (".extractor." in n or ".reference_" in n)]
    assert frozen
    for name in frozen:
        assert torch.equal(first["objective"][name], final["objective"][name]), name
    assert final["config"]["static"]["enabled"]
    assert final["config"]["train"]["grad_clip"] == 0
    if group == "A":
        assert final["config"]["method"] == "fd_only" and "critic_optimizer" not in final
        assert all("adv_fd" not in r for r in rows)
        assert not any(n.startswith("extractor.") for n in final["objective"])
    else:
        assert group == "B" and final["config"]["method"] == "advfd_reconstruction"
        assert [r["critic_updates"] for r in rows] == [0, 0, 1, 1, 2, 2]
        assert [r["adv_statistics_updates"] for r in rows] == [
            {"real": n, "fake": n} for n in (0, 0, 1, 2, 3, 4)]
        assert [r["adv_schedule_step"] for r in rows] == list(range(6))
        assert [r["adv_active"] for r in rows] == [False, False, True, True, True, True]
        assert [r["adv_effective_weight"] for r in rows] == [0., 0., 0., .05, .1, .1]
        assert all(math.isfinite(r["adv_fd"]) for r in rows[2:])
        assert all(math.isfinite(r["critic_grad_norm"]) and r["critic_grad_norm"] > 0
                   for r in rows if r["critic_updated"])
        manifest = json.loads((run / "advfd_parameters.json").read_text())
        assert manifest["scope"] == ("full" if label == "real" else "full_engineering")
        assert manifest["trainable_parameters"] == manifest["total_parameters"]
        assert final["config"]["adaptive"]["trainable_scope"] == "paper"
        assert not bool(first["objective"]["fake_statistics.initialized"])
        assert bool(final["objective"]["fake_statistics.initialized"])
        # No dynamic parameter changes before activation (two completed G steps).
        before = read_checkpoint(run / "checkpoints/step_0000002.pt")
        for name in manifest["trainable_names"]:
            key = "extractor." + name
            assert torch.equal(first["objective"][key], before["objective"][key]), key
        assert any(not torch.equal(first["objective"]["extractor." + n], final["objective"]["extractor." + n])
                   for n in manifest["trainable_names"])
        for name, value in final["objective"].items():
            if name.startswith("extractor.") and any(s in name for s in ("running_mean", "running_var", "num_batches_tracked")):
                assert torch.equal(first["objective"][name], value), name
        traces = []
        for stop in (2, 4, 6):
            traces.extend(json.loads((root / f"{label}_to_{stop}.trace.json").read_text()))
        assert [r["schedule_step"] for r in traces] == list(range(6))
        assert all(r["gradient_scopes_checked"] for r in traces)
    return {"encoder_decoder_updated": True, "static_extractor_and_reference_unchanged": True,
            "statistics_counts_verified": True, "resume_data_order_verified": True,
            "B_D_then_G_and_gradient_isolation_verified": group == "B"}


def verify_evaluation(folder, step, names):
    import csv
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["engineering_only"] and metrics["num_samples"] == 32
    assert metrics["checkpoint_step"] == step
    assert set(metrics["fd"]) == set(names) and all(math.isfinite(v) for v in metrics["fd"].values())
    assert set(metrics["paired"]) == {"psnr", "ssim"}
    assert all(math.isfinite(v) for v in metrics["paired"].values())
    export = json.loads((folder / "reconstructions/complete.json").read_text())
    assert len(export["sample_ids"]) == len(set(export["sample_ids"])) == export["count"] == 32
    assert len(list((folder / "reconstructions").glob("*.png"))) == 32
    with (folder / "per_image.csv").open() as handle:
        per_image = list(csv.DictReader(handle))
    assert [r["sample_id"] for r in per_image] == export["sample_ids"]
    for row in per_image:
        assert all(math.isfinite(float(v)) for k, v in row.items() if k != "sample_id")
    return {"fd": metrics["fd"], "paired": metrics["paired"], "sample_ids": export["sample_ids"],
            "representations": metrics["representations"]}


def require_allocation():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use jobs/ab_acceptance_nibi.sbatch; no ML on login nodes")


def worker():
    require_allocation()
    import torch
    from recon_fd.cli import train_main, evaluate_main
    from recon_fd.engine import trainer
    from recon_fd.provenance import write_json
    from recon_fd.runtime import determinism_settings
    stage, root, *arguments = sys.argv[2:]
    root = Path(root)
    records = []
    trainer.adversarial_g_step = audited_b_step(trainer.adversarial_g_step, records)
    torch.cuda.reset_peak_memory_stats()
    started, succeeded = time.monotonic(), False
    try:
        (evaluate_main if stage.startswith("evaluate") else train_main)(arguments)
        succeeded = True
    finally:
        torch.cuda.synchronize()
        write_json(root / f"{stage}.trace.json", records)
        write_json(root / f"{stage}.resources.json", {
            "succeeded": succeeded, "seconds": time.monotonic() - started,
            "determinism": determinism_settings(),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})


def group_main(group, root):
    require_allocation()
    from recon_fd.provenance import write_json
    root.mkdir(parents=True, exist_ok=False)
    started, stages, active = time.monotonic(), [], "configuration"
    try:
        def run(stage, arguments):
            nonlocal active
            active = stage
            print(f"START {group}/{stage}", flush=True)
            log = root / f"{stage}.log"
            with log.open("w") as handle:
                result = subprocess.run([sys.executable, "-u", __file__, "worker", stage, str(root), *arguments],
                                        stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text()[-10000:], flush=True)
                raise RuntimeError(f"{stage}: exit {result.returncode}")
            stages.append(stage)
            print(f"PASS {group}/{stage}", flush=True)

        comparisons, checks, evaluations = {}, {}, {}
        configs = {"A": ("smoke.yaml", "fd_only_acceptance_nibi.yaml"),
                   "B": ("advfd_smoke.yaml", "advfd_acceptance_nibi.yaml")}[group]
        for label, filename in zip(("tiny", "real"), configs):
            common = ["--config", f"configs/{filename}", "--set", "runtime.device=cuda",
                      "--set", "train.save_every=2", "--set", "train.log_every=1",
                      "--set", f"static.reference_cache={root.parent / 'cache' / label}"]
            comparisons[label] = protocol(root, label, common, run)
            active = f"verify_{label}"
            checks[label] = verify_run(root, group, label)
            names = ["inception"] if label == "real" else (["tiny", "tiny_holdout"])
            evaluated = {}
            for step in (0, 6):
                folder = root / f"{label}_evaluation_step{step}"
                run(f"evaluate_{label}_{step}", [*common, "--checkpoint",
                    str(root / f"{label}_resumed/checkpoints/step_{step:07d}.pt"), "--output", str(folder)])
                active = f"verify_evaluate_{label}_{step}"
                evaluated[str(step)] = verify_evaluation(folder, step, names)
            assert evaluated["0"]["sample_ids"] == evaluated["6"]["sample_ids"]
            assert evaluated["0"]["representations"] == evaluated["6"]["representations"]
            evaluations[label] = evaluated
        resources = {p.stem: json.loads(p.read_text()) for p in root.glob("*.resources.json")}
        assert len(resources) == len(stages) and all(v["succeeded"] for v in resources.values())
        assert all(v["determinism"]["algorithms"] and not v["determinism"]["warn_only"]
                   for v in resources.values())
        write_json(root / "result.json", {"status": "passed", "group": group, "engineering_only": True,
            "git_commit": os.environ["FD_EXPECTED_COMMIT"], "job_id": os.environ["SLURM_JOB_ID"],
            "steps": 6, "train_pool_images": 128, "evaluation_images_per_checkpoint": 32,
            "seconds": time.monotonic() - started, "stages": stages, "resume_comparisons": comparisons,
            "checks": checks, "evaluations": evaluations, "resources": resources,
            "boundary": "Single Inception path, shortened B schedule, no 50k/SIM/LPIPS or visual-quality claim."})
    except Exception as error:
        write_json(root / "result.json", {"status": "failed", "group": group, "engineering_only": True,
            "stage": active, "stages_passed": stages, "error": repr(error)})
        raise


def selected_group():
    group = os.environ.get("FD_ACCEPT_GROUP")
    if group not in {"A", "B"}:
        raise ValueError("Set FD_ACCEPT_GROUP to exactly A or B; submit each group separately")
    return group


def main():
    require_allocation()
    group_to_run = selected_group()
    import torch
    from nibi_smoke import prepare_images
    from recon_fd.provenance import implementation_fingerprint, write_json
    root = Path(os.environ["FD_AB_SMOKE_ROOT"])
    root.mkdir(parents=True, exist_ok=False)
    started, active, groups = time.monotonic(), "preflight", {}
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA GPU required")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if commit != os.environ["FD_EXPECTED_COMMIT"]:
            raise RuntimeError("Checkout changed after submission")
        for diff in (["git", "diff", "--quiet"], ["git", "diff", "--cached", "--quiet"]):
            subprocess.run(diff, check=True)
        tokenizer = Path(os.environ["TOKENIZER_CHECKPOINT"])
        for path in (tokenizer / "config.json", tokenizer / "diffusion_pytorch_model.safetensors",
                     Path(os.environ["INCEPTION_WEIGHTS"])):
            if not path.is_file():
                raise FileNotFoundError(path)
        write_json(root / "environment.json", {"git_commit": commit, "job_id": os.environ["SLURM_JOB_ID"],
            "implementation_sha256": implementation_fingerprint(), "engineering_only": True,
            "python": sys.version, "hostname": os.uname().nodename, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {n: importlib.metadata.version(n) for n in
                         ("torch", "torchvision", "timm", "diffusers", "numpy", "scikit-image")}})
        write_json(root / "image_manifest.json", {"engineering_only": True,
            "samples": prepare_images(Path(os.environ["IMAGENET_ROOT"]), root / "images")})
        for group in (group_to_run,):
            active = group
            # Separate process releases previous group's model/checkpoint memory.
            result = subprocess.run([sys.executable, "-u", __file__, "group", group, str(root / group)])
            path = root / group / "result.json"
            report = json.loads(path.read_text()) if path.is_file() else {"status": "failed", "error": "No result file"}
            groups[group] = {"status": report["status"], "exit_code": result.returncode, "result": str(path)}
            if result.returncode:
                groups[group]["status"] = "failed"
        status = "passed" if all(g["status"] == "passed" for g in groups.values()) else "failed"
        write_json(root / "result.json", {"status": status, "engineering_only": True, "groups": groups,
            "git_commit": commit, "job_id": os.environ["SLURM_JOB_ID"], "seconds": time.monotonic() - started,
            "C": "Separate current-both job; this result does not certify C or 50k experiments."})
        if status != "passed":
            raise SystemExit(1)
        print(f"{group_to_run} ALL CHECKS PASSED: {root / 'result.json'}", flush=True)
    except Exception as error:
        write_json(root / "result.json", {"status": "failed", "engineering_only": True,
            "stage": active, "groups": groups, "error": repr(error)})
        raise


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker()
    elif sys.argv[1:2] == ["group"] and len(sys.argv) == 4 and sys.argv[2] in {"A", "B"}:
        group_main(sys.argv[2], Path(sys.argv[3]))
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError("Expected no arguments, worker, or group A|B PATH")
