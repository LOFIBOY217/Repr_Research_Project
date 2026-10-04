"""C integration on one allocated GPU. Never execute ML on a login node."""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from nibi_smoke import prepare_images, worker as common_worker


def compare_states(left, right):
    """GPU comparisons allow rounding; report whether they were bitwise exact."""
    import torch
    bitwise = True
    def compare(a, b):
        nonlocal bitwise
        if isinstance(a, torch.Tensor):
            bitwise = bitwise and torch.equal(a, b)
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b
    for field in ("model", "objective", "optimizer", "critic_optimizer"):
        compare(left[field], right[field])
    return {"passed": True, "bitwise_equal": bitwise, "rtol": 1e-5, "atol": 1e-7}


def main(current_both=False):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Submit jobs/ours_smoke_nibi.sbatch; login-node execution is prohibited")
    import torch
    from recon_fd.engine.checkpoint import read_checkpoint
    from recon_fd.provenance import implementation_fingerprint, write_json
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    root = Path(os.environ["FD_C_SMOKE_ROOT"])
    root.mkdir(parents=True, exist_ok=False)
    stages, active_stage = [], "preflight"
    started = time.monotonic()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one visible CUDA GPU required")
        tokenizer = Path(os.environ["TOKENIZER_CHECKPOINT"])
        for path in (tokenizer / "config.json", tokenizer / "diffusion_pytorch_model.safetensors",
                     Path(os.environ["INCEPTION_WEIGHTS"])):
            if not path.is_file():
                raise FileNotFoundError(path)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if commit != os.environ["FD_EXPECTED_COMMIT"]:
            raise RuntimeError("Checkout changed after submission")
        write_json(root / "environment.json", {
            "engineering_only": True, "job_id": os.environ["SLURM_JOB_ID"], "git_commit": commit,
            "variant": "current_both" if current_both else "current_real_fake_ema",
            "implementation_sha256": implementation_fingerprint(), "hostname": os.uname().nodename,
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "python": sys.version,
            "packages": {n: importlib.metadata.version(n) for n in
                         ("torch", "torchvision", "timm", "diffusers", "numpy", "scikit-image")}})
        write_json(root / "image_manifest.json", {"engineering_only": True,
                   "samples": prepare_images(Path(os.environ["IMAGENET_ROOT"]), root / "images")})

        def run(stage, arguments):
            nonlocal active_stage
            active_stage = stage
            print(f"START {stage}", flush=True)
            path = root / f"{stage}.log"
            with path.open("w") as handle:
                result = subprocess.run([sys.executable, "-u", __file__, "worker", stage, str(root), *arguments],
                                        stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                print(path.read_text()[-10000:], flush=True)
                raise RuntimeError(f"{stage} failed with exit {result.returncode}")
            stages.append(stage)
            print(f"PASS {stage}", flush=True)

        comparisons = {}
        protocols = [("tiny", []), ("real", [])] if current_both else [
            ("tiny", []), ("tiny_ema", ["--set", "method=ours_real_ema", "--set", "adaptive.real_stats.mode=ema"]), ("real", [])]
        real_config = "configs/ours_current_both_nibi_smoke.yaml" if current_both else "configs/ours_nibi_smoke.yaml"
        tiny_config = "configs/ours_current_both_smoke.yaml" if current_both else "configs/ours_smoke.yaml"
        for label, extra in protocols:
            config = real_config if label == "real" else tiny_config
            common = ["--config", config, "--set", "runtime.device=cuda", *extra]
            a, b = root / f"{label}_resumed", root / f"{label}_continuous"
            run(f"{label}_train", [*common, "--set", f"train.output={a}", "--set", "train.steps=2"])
            run(f"{label}_resume", [*common, "--set", f"train.output={a}", "--set", "train.steps=4", "--resume",
                                   str(a / "checkpoints/step_0000002.pt")])
            run(f"{label}_continuous", [*common, "--set", f"train.output={b}", "--set", "train.steps=4"])
            active_stage = f"{label}_resume_comparison"
            left = read_checkpoint(a / "checkpoints/step_0000004.pt")
            right = read_checkpoint(b / "checkpoints/step_0000004.pt")
            comparisons[label] = compare_states(left, right)
            del left, right

        for step in (0, 4):
            run(f"evaluate_real_{step}", ["--config", real_config, "--checkpoint",
                str(root / f"real_resumed/checkpoints/step_{step:07d}.pt"),
                "--output", str(root / f"evaluation_step{step}")])

        active_stage = "verify_current_reference"
        initial = read_checkpoint(root / "real_resumed/checkpoints/step_0000000.pt")
        final = read_checkpoint(root / "real_resumed/checkpoints/step_0000004.pt")
        rows = [json.loads(line) for line in (root / "real_resumed/train.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == [1, 2, 3, 4]
        assert [r["critic_updates"] for r in rows] == [1, 1, 2, 2]
        assert [r["fake_ema_updates"] for r in rows] == ([0, 0, 0, 0] if current_both else [1, 2, 3, 4])
        assert [r["real_reference_refreshes"] for r in rows] == [2, 2, 3, 3]
        for row in rows:
            assert not row["static_enabled"] and row["raw_fd"] == {}
            assert row["loss_psi_version"] == row["real_feature_version"]
            assert row["real_reference_current_psi"] and row["real_reference_pool_samples"] == 128
            assert math.isfinite(row["adv_fd"])
            assert all(math.isfinite(v) and v > 0 for v in row["group_grad_norm"].values())
            if current_both:
                assert row["fake_feature_psi_version"] == row["loss_psi_version"]
                assert row["fake_feature_generator_version"] == row["loss_generator_version"] == row["step"]-1
                assert row["generator_updates"] == row["step"] and not row["fake_cache_current_after_G"]
                assert not row["fake_statistics_historical"] and row["real_ema_updates"] == 0
                assert row["fd_batch_size"] == row["fake_reference_pool_samples"] == 128
                assert row["gradient_estimator"] == "full_pool_two_pass_chain_rule"
        if current_both:
            assert [r["fake_reference_refreshes"] for r in rows] == [2, 3, 5, 6]
            assert [r["gradient_feature_images"] for r in rows] == [256, 384, 640, 768]
            assert not any("ema" in n for n in final["objective"])
        for group in ("encoder", "decoder"):
            assert any(not torch.equal(initial["model"][n], p) for n, p in final["model"].items()
                       if f"{group}." in n), group
        assert not any(n.startswith("static.") for n in final["objective"])
        assert any(not torch.equal(initial["objective"][n], p) for n, p in final["objective"].items()
                   if n.startswith("extractor.")), "Dynamic extractor did not update"
        for n, p in final["objective"].items():
            if n.startswith("extractor.") and any(s in n for s in ("running_mean", "running_var", "num_batches_tracked")):
                assert torch.equal(p, initial["objective"][n]), n
        manifest = json.loads((root / "real_resumed/advfd_parameters.json").read_text())
        assert manifest["trainable_parameters"] == manifest["total_parameters"] and manifest["scope"] == "full"

        # Re-encode the real pool independently with checkpoint psi, not its cached features.
        from recon_fd.representations import build_representation
        from recon_fd.data import build_dataset, selected_dataset, sequential_loader
        from recon_fd.objectives.statistics import RunningMoments
        config = final["config"]
        extractor = build_representation(config["adaptive"]["representation"]).cuda()
        extractor.load_state_dict({n.removeprefix("extractor."): p for n, p in final["objective"].items()
                                   if n.startswith("extractor.")})
        real = config["adaptive"]["real_stats"]
        dataset = selected_dataset(build_dataset(config, "train"), real["samples"], real["seed"], True)
        moments = RunningMoments()
        with torch.no_grad():
            for batch in sequential_loader(dataset, real["batch_size"], 0):
                moments.update(extractor(batch["image"].cuda()))
        current = moments.moments()
        torch.testing.assert_close(current.mean.cpu(), final["objective"]["real_reference.mean"], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(current.cov.cpu(), final["objective"]["real_reference.covariance"], rtol=1e-5, atol=1e-7)
        if current_both:
            active_stage = "verify_current_reconstruction_reference"
            from recon_fd.tokenizers import build_tokenizer
            from recon_fd.objectives.current_both import ReconstructionReference
            tokenizer_model = build_tokenizer(config["tokenizer"]).cuda().eval()
            tokenizer_model.load_state_dict(final["model"])
            cache = ReconstructionReference(extractor.dimension).cuda()
            cache.load_state_dict({n.removeprefix("fake_reference."): p for n, p in final["objective"].items()
                                   if n.startswith("fake_reference.")})
            psi_version = int(final["objective"]["critic_updates"])
            generator_version = int(final["objective"]["generator_updates"])
            # Saved after G: pre-G cache is deliberately marked stale, never reused.
            assert not cache.current(psi_version, generator_version)
            refreshes = int(cache.refreshes)
            cache.refresh(tokenizer_model, extractor, dataset, config["train"]["batch_size"], 0,
                          psi_version, generator_version)
            assert cache.current(psi_version, generator_version) and int(cache.refreshes) == refreshes + 1
            direct = RunningMoments()
            with torch.no_grad():
                for batch in sequential_loader(dataset, config["train"]["batch_size"], 0):
                    direct.update(extractor(tokenizer_model(batch["image"].cuda())))
            moments = direct.moments()
            torch.testing.assert_close(moments.mean, cache.mean, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(moments.cov, cache.covariance, rtol=1e-5, atol=1e-7)
        evaluations = {}
        for step in (0, 4):
            folder = root / f"evaluation_step{step}"
            metrics = json.loads((folder / "metrics.json").read_text())
            assert metrics["engineering_only"] and metrics["num_samples"] == 32
            assert len(list((folder / "reconstructions").glob("*.png"))) == 32
            assert metrics["checkpoint_step"] == step and math.isfinite(metrics["fd"]["inception"])
            evaluations[str(step)] = {"fd": metrics["fd"], "paired": metrics["paired"]}
        resources = {p.stem: json.loads(p.read_text()) for p in root.glob("*.resources.json")}
        write_json(root / "result.json", {
            "status": "passed", "engineering_only": True, "job_id": os.environ["SLURM_JOB_ID"],
            "git_commit": commit, "stages": stages, "seconds": time.monotonic() - started,
            "resume_comparisons": comparisons, "current_psi_real_reference_verified": True,
            "variant": "current_both" if current_both else "current_real_fake_ema",
            "current_model_reconstruction_reference_verified": current_both,
            "encoder_decoder_and_psi_updated": True, "no_static_branch": True,
            "real_reference_images": 128, "evaluation_images": 32, "steps": 4,
            "evaluations": evaluations, "resources": resources,
            "boundary": ("Engineering only, not 50k validation or evidence of reduced hacking; full-pool steps are not minibatch steps."
                         if current_both else "Engineering only, not 50k validation or evidence of reduced hacking; fake EMA still historical.")})
        print(f"ALL CHECKS PASSED: {root / 'result.json'}", flush=True)
    except Exception as error:
        write_json(root / "result.json", {"status": "failed", "engineering_only": True,
                   "stage": active_stage, "stages_passed": stages, "error": repr(error)})
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Worker requires a SLURM allocation")
        import torch
        torch.backends.cudnn.deterministic = True
        common_worker()
    else:
        if sys.argv[1:] not in ([], ["current_both"]):
            raise ValueError("Expected no argument or current_both")
        main(current_both=sys.argv[1:] == ["current_both"])
