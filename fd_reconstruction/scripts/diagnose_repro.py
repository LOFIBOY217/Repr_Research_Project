"""Read-only old-run analysis and isolated CUDA reproducibility experiments.

Does not modify training algorithms, checkpoints, or acceptance tolerances.
Strict determinism is an experimental intervention confined to this process.
"""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def difference(a, b, path="root"):
    import numpy as np
    import torch
    rows = []
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return [{"path": path, "kind": "shape_or_dtype"}]
        if not torch.equal(a, b):
            if a.is_floating_point():
                delta = (a.double() - b.double()).abs()
                rows.append({"path": path, "kind": "tensor", "numel": a.numel(),
                    "unequal": int((a != b).sum()), "max_abs": float(delta.max()),
                    "relative_l2": float(delta.norm() / a.double().norm().clamp_min(1e-30)),
                    "outside_original_tolerance": int((~torch.isclose(a, b, rtol=1e-5, atol=1e-7)).sum())})
            else:
                rows.append({"path": path, "kind": "integer_tensor", "unequal": int((a != b).sum())})
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            rows.append({"path": path, "kind": "numpy_array"})
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            rows.append({"path": path, "kind": "keys"})
        for key in a.keys() & b.keys():
            rows.extend(difference(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            rows.append({"path": path, "kind": "length"})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                rows.extend(difference(x, y, f"{path}.{i}"))
    elif a != b:
        rows.append({"path": path, "kind": "value", "left": str(a), "right": str(b)})
    return rows


def checkpoint_differences(root):
    from recon_fd.engine.checkpoint import read_checkpoint
    from recon_fd.provenance import write_json
    reports = {}
    for group, parent, end in (("A", Path(os.environ["FD_AB_SOURCE"]) / "A", 6),
                               ("B", Path(os.environ["FD_AB_SOURCE"]) / "B", 6),
                               ("C", Path(os.environ["FD_C_SOURCE"]), 4)):
        reports[group] = {}
        for step in range(0, end + 1, 2):
            left = read_checkpoint(parent / f"real_resumed/checkpoints/step_{step:07d}.pt")
            right = read_checkpoint(parent / f"real_continuous/checkpoints/step_{step:07d}.pt")
            fields = ["model", "objective", "optimizer", "rng", "data_identity", "signature"]
            if "critic_optimizer" in left:
                fields.append("critic_optimizer")
            report = {field: difference(left[field], right[field], field) for field in fields}
            reports[group][str(step)] = {field: {"differing_leaves": len(rows),
                "outside_tolerance_leaves": sum(r.get("outside_original_tolerance", 1) > 0 for r in rows),
                "largest": sorted(rows, key=lambda r: r.get("max_abs", 0), reverse=True)[:8]}
                for field, rows in report.items()}
            write_json(root / f"checkpoint_{group}_{step}.json", report)
            print("CHECKPOINT", group, step, json.dumps(reports[group][str(step)]), flush=True)
            del left, right, report
            gc.collect()
    write_json(root / "checkpoint_summary.json", reports)


def repeat_vjp(function, targets, repeats=4):
    """Repeat the entire forward/backward at identical weights and inputs."""
    import torch
    first_value, first_grads, differences = None, None, []
    for iteration in range(repeats):
        output = function()
        upstream = torch.sin(torch.arange(output.numel(), device=output.device, dtype=torch.float32) + 1)
        upstream = upstream.reshape(output.shape).to(output.dtype)
        gradients = torch.autograd.grad(output, list(targets.values()), grad_outputs=upstream)
        value = output.detach().cpu()
        grads = {name: gradient.detach().cpu() for name, gradient in zip(targets, gradients)}
        if iteration == 0:
            first_value, first_grads = value, grads
        else:
            differences.append({"forward": difference(first_value, value),
                                "backward": difference(first_grads, grads)})
        del output, gradients
    return {"repeats": repeats,
            "forward_bitwise_equal": all(not r["forward"] for r in differences),
            "backward_bitwise_equal": all(not r["backward"] for r in differences),
            "differences": differences}


def component_probes(root):
    import torch
    from recon_fd.data import build_dataset, ResumableBatchSampler
    from recon_fd.engine.checkpoint import read_checkpoint
    from recon_fd.engine.trainer import build_objective
    from recon_fd.tokenizers import build_tokenizer
    from recon_fd.representations.inception import resize_tf
    from recon_fd.provenance import write_json
    state = read_checkpoint(Path(os.environ["FD_AB_SOURCE"]) / "A/real_resumed/checkpoints/step_0000000.pt")
    config = state["config"]
    dataset = build_dataset(config, "train")
    indices = next(iter(ResumableBatchSampler(len(dataset), 2, config["data"]["seed"], 0, 1)))
    images = torch.stack([dataset[i]["image"] for i in indices]).cuda()
    model = build_tokenizer(config["tokenizer"]).cuda().eval()
    model.load_state_dict(state["model"])
    objective, _ = build_objective(config, dataset, "cuda")
    objective.load_state_dict(state["objective"])
    space = objective.spaces["inception"]
    with torch.no_grad():
        reconstruction = model(images)
        fixed_features = space.extractor(reconstruction)
    del state
    gc.collect()
    report = {}
    # Same tensors and weights for both settings, no optimizer/state commits.
    pixel = reconstruction.detach().clone().requires_grad_(True)
    features = fixed_features.detach().clone().requires_grad_(True)
    named = dict(model.named_parameters())
    selected = {n: named[n] for n in ("model.encoder.conv_in.weight", "model.decoder.conv_in.weight",
                                     "model.decoder.conv_out.weight")}
    for mode in ("baseline", "strict"):
        torch.use_deterministic_algorithms(mode == "strict")
        report[mode] = {}
        probes = [
            ("resize_tf", lambda: resize_tf(pixel), {"input": pixel}),
            ("inception_input", lambda: space.extractor(pixel), {"input": pixel}),
            ("vae_parameters_eval", lambda: model(images), selected),
            ("official_fd_features", lambda: space.statistics.fd(features, space.reference_mean,
                space.reference_cov, space.reference_sqrt), {"features": features}),
            ("official_fd_pixels", lambda: objective(pixel).loss, {"input": pixel}),
            ("end_to_end_parameters", lambda: objective(model(images)).loss, selected),
        ]
        for name, function, targets in probes:
            started = time.monotonic()
            try:
                report[mode][name] = repeat_vjp(function, targets)
            except Exception as error:
                report[mode][name] = {"error": repr(error), "traceback": traceback.format_exc()}
            report[mode][name]["seconds"] = time.monotonic() - started
            write_json(root / "component_probes.json", report)
            print("PROBE", mode, name, json.dumps(report[mode][name]), flush=True)
        # Record the dispatched backward operators, not guessed library defaults.
        try:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                loss = objective(model(images)).loss
                torch.autograd.grad(loss, list(selected.values()))
            markers = ("attention", "index", "scatter", "upsample", "grid_sampler")
            report[mode]["dispatched_operators"] = [
                {"name": event.key, "count": event.count} for event in prof.key_averages()
                if any(s in event.key.lower() for s in markers)]
        except Exception as error:
            report[mode]["profiler_error"] = repr(error)
        # A/B use training mode with checkpointing, unlike C's eval-with-grad.
        model.train()
        try:
            report[mode]["vae_parameters_train_checkpointed"] = repeat_vjp(lambda: model(images), selected)
        except Exception as error:
            report[mode]["vae_parameters_train_checkpointed"] = {"error": repr(error), "traceback": traceback.format_exc()}
        model.eval()
        write_json(root / "component_probes.json", report)
        print("PROBE", mode, "vae_parameters_train_checkpointed",
              json.dumps(report[mode]["vae_parameters_train_checkpointed"]), flush=True)
    return report


def strict_training(root):
    """New isolated 2->4 vs continuous-4 runs, leaving old evidence untouched."""
    from recon_fd.engine.checkpoint import read_checkpoint
    from recon_fd.provenance import write_json
    from ab_acceptance import compare_checkpoints
    reports = {}
    for group, source in (("A", Path(os.environ["FD_AB_SOURCE"]) / "A"),
                          ("B", Path(os.environ["FD_AB_SOURCE"]) / "B"),
                          ("C", Path(os.environ["FD_C_SOURCE"]))):
        folder = root / f"strict_{group}"
        folder.mkdir()
        # Preserve original resolved configuration, data, and learning rates.
        config = json.loads((source / "real_resumed/config.json").read_text())
        write_json(folder / "config.json", config)
        reports[group] = {"status": "running", "stages": []}
        try:
            for label, end, resume in (("train", 2, False), ("resume", 4, True), ("continuous", 4, False)):
                run = folder / ("continuous" if label == "continuous" else "resumed")
                args = ["--config", str(folder / "config.json"), "--set", f"train.output={run}",
                        "--set", f"train.steps={end}", "--set", "train.save_every=2"]
                if resume:
                    args += ["--resume", str(run / "checkpoints/step_0000002.pt")]
                with (folder / f"{label}.log").open("w") as handle:
                    result = subprocess.run([sys.executable, "-u", __file__, "strict-worker", *args],
                                            stdout=handle, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{label} exit {result.returncode}: " + (folder / f"{label}.log").read_text()[-7000:])
                reports[group]["stages"].append(label)
                print("STRICT TRAIN", group, label, "passed", flush=True)
            reports[group]["comparisons"] = {}
            for step in (0, 2, 4):
                reports[group]["comparisons"][str(step)] = compare_checkpoints(
                    read_checkpoint(folder / f"resumed/checkpoints/step_{step:07d}.pt"),
                    read_checkpoint(folder / f"continuous/checkpoints/step_{step:07d}.pt"))
            reports[group]["status"] = "passed"
        except Exception as error:
            reports[group].update(status="failed", error=repr(error), traceback=traceback.format_exc())
        write_json(root / "strict_training.json", reports)
        print("STRICT RESULT", group, json.dumps(reports[group]), flush=True)
    return reports


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Diagnosis requires a SLURM allocation; no login-node ML")
    import torch
    from recon_fd.provenance import implementation_fingerprint, write_json
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if sys.argv[1:2] == ["strict-worker"]:
        torch.use_deterministic_algorithms(True)
        from recon_fd.cli import train_main
        train_main(sys.argv[2:])
        return
    if sys.argv[1:]:
        raise ValueError("Expected no argument or strict-worker")
    root = Path(os.environ["FD_DIAG_ROOT"])
    root.mkdir(parents=True, exist_ok=False)
    stage = "preflight"
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU required")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        assert commit == os.environ["FD_EXPECTED_COMMIT"]
        write_json(root / "environment.json", {"git_commit": commit, "implementation_sha256": implementation_fingerprint(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "hostname": os.uname().nodename, "job_id": os.environ["SLURM_JOB_ID"],
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "diagnosis_only": True})
        stage = "checkpoint_differences"
        checkpoint_differences(root)
        stage = "component_probes"
        probes = component_probes(root)
        gc.collect()
        torch.cuda.empty_cache()
        stage = "strict_training"
        training = strict_training(root)
        write_json(root / "result.json", {"status": "diagnosis_completed", "diagnosis_only": True,
            "component_probes": probes, "strict_training": training,
            "boundary": "Experimental flags only; not an algorithm fix, new acceptance, or 50k result."})
    except Exception as error:
        write_json(root / "result.json", {"status": "failed", "diagnosis_only": True,
            "stage": stage, "error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
