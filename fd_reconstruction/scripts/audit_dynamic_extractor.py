"""Audit actual dynamic-feature updates in trusted reconstruction checkpoints.

Run on a compute node: this loads our own pickle checkpoints and pretrained models.
EMA/reference buffers are deliberately excluded from parameter-change evidence.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from recon_fd.config import validate
from recon_fd.data import build_dataset, selected_dataset
from recon_fd.engine.checkpoint import read_checkpoint
from recon_fd.provenance import write_json
from recon_fd.representations import build_representation
from recon_fd.representations.lora import apply_qkv_lora
from recon_fd.tokenizers import build_tokenizer


def extractor_state(checkpoint):
    prefix = "extractor."
    return {name[len(prefix):]: value for name, value in checkpoint["objective"].items()
            if name.startswith(prefix)}


def static_anchors(checkpoint):
    return {name: value for name, value in checkpoint["objective"].items()
            if name.startswith("static.spaces.") and (".extractor." in name or ".reference_" in name)}


def feature_spec(config):
    if config["method"] == "advfd_reconstruction":
        name = config["adaptive"]["representation"]
        return next(spec for spec in config["static"]["representations"] if spec["name"] == name)
    if config["method"] == "ours_current_both":
        return config["adaptive"]["representation"]
    raise ValueError("Audit supports MAE-B or exact-pool MAE-C only")


def build_dynamic_extractor(config, state, device):
    spec = feature_spec(config)
    extractor = build_representation(spec)
    if config["method"] == "advfd_reconstruction" and spec["kind"] == "timm":
        lora = config["adaptive"]["lora"]
        apply_qkv_lora(extractor, lora["rank"], lora["alpha"], lora["dropout"])
    extractor.load_state_dict(state, strict=True)
    return extractor.to(device).eval().requires_grad_(False)


def fixed_probe(config, checkpoint_zero, device, count):
    dataset = build_dataset(config, "val")
    count = min(count, len(dataset))
    probe = selected_dataset(dataset, count, config["data"]["seed"] + 991, True)
    images = torch.stack([probe[index]["image"] for index in range(count)]).to(device)
    model = build_tokenizer(config["tokenizer"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint_zero["model"], strict=True)
    with torch.no_grad():
        reconstruction = torch.cat([model(chunk) for chunk in images.split(2)]).detach()
    del model
    return images, reconstruction, probe.ids


@torch.no_grad()
def probe_features(extractor, images, reconstruction):
    def encode(batch):
        return torch.cat([extractor(chunk).detach().float().cpu() for chunk in batch.split(2)])
    return encode(images), encode(reconstruction)


def parameter_difference(before, after, trainable):
    if before.keys() != after.keys() or not trainable <= before.keys():
        raise ValueError("Extractor parameter keys changed across checkpoints")
    changed, unexpected = [], []
    delta_squared = norm_squared = 0.0
    max_absolute = 0.0
    for name in before:
        old, new = before[name], after[name]
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError(f"Extractor tensor schema changed: {name}")
        if name not in trainable:
            if not torch.equal(old, new):
                unexpected.append(name)
            continue
        if not torch.isfinite(new).all():
            raise FloatingPointError(f"Non-finite trainable extractor tensor: {name}")
        if not torch.equal(old, new):
            changed.append(name)
        difference = new.double() - old.double()
        delta_squared += difference.square().sum().item()
        norm_squared += old.double().square().sum().item()
        max_absolute = max(max_absolute, difference.abs().max().item())
    return {"changed_trainable_tensors": len(changed), "trainable_tensor_count": len(trainable),
            "relative_l2_delta": math.sqrt(delta_squared) / max(math.sqrt(norm_squared), 1e-12),
            "max_abs_delta": max_absolute, "unexpected_frozen_changes": unexpected,
            "changed_trainable_names": changed}


def audit_run(run_dir, device="cuda", probe_count=8, require_mae=True):
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / "train.jsonl").read_text().splitlines()]
    if len(rows) < 2 or [row["step"] for row in rows[:2]] != [1, 2]:
        raise ValueError("Need completed preflight steps 1 and 2")
    manifest = json.loads((run_dir / "advfd_parameters.json").read_text())
    checkpoints = [read_checkpoint(run_dir / f"checkpoints/step_{step:07d}.pt") for step in range(3)]
    config = checkpoints[0]["config"]
    validate(config)
    spec = feature_spec(config)
    if require_mae and (spec.get("name") != "mae" or spec.get("model_name") != "vit_large_patch16_224.mae"):
        raise ValueError("Expected the actual MAE dynamic backbone")
    if config["method"] == "ours_current_both":
        if config["adaptive"]["fake_stats"] != {"mode": "reencode_pool", "gradient": "full_pool_replay"}:
            raise ValueError("C audit requires exact paired-pool statistics")
        if config["adaptive"]["real_stats"]["mode"] != "reencode_pool":
            raise ValueError("C audit requires current real-pool statistics")
        if manifest["scope"] != "full" or manifest["trainable_parameters"] != manifest["total_parameters"]:
            raise ValueError("C MAE is not fully trainable")
    else:
        if config["adaptive"]["real_stats"]["mode"] != "ema":
            raise ValueError("B audit requires dynamic real EMA")
        if require_mae and manifest["scope"] != "lora":
            raise ValueError("B MAE is expected to use LoRA")
    trainable = set(manifest["trainable_names"])
    if not trainable:
        raise ValueError("No trainable dynamic extractor parameters")
    if require_mae and config["method"] == "advfd_reconstruction" and not all(".lora_" in name for name in trainable):
        raise ValueError("B has non-LoRA trainable parameters")
    optimizer_count = sum(len(group["params"]) for group in checkpoints[0]["critic_optimizer"]["param_groups"])
    if optimizer_count != len(trainable):
        raise ValueError("Critic optimizer parameter count differs from manifest")
    if any(checkpoint["step"] != step or checkpoint["signature"] != checkpoints[0]["signature"]
           for step, checkpoint in enumerate(checkpoints)):
        raise ValueError("Checkpoint steps or training signatures differ")

    real, fake, sample_ids = fixed_probe(config, checkpoints[0], device, probe_count)
    extractor = build_dynamic_extractor(config, extractor_state(checkpoints[0]), device)
    features = []
    for checkpoint in checkpoints:
        extractor.load_state_dict(extractor_state(checkpoint), strict=True)
        features.append(probe_features(extractor, real, fake))
    repeat_real, repeat_fake = probe_features(extractor, real, fake)
    repeat_noise = max((repeat_real - features[-1][0]).double().norm().item(),
                       (repeat_fake - features[-1][1]).double().norm().item())

    reports = []
    for step in (1, 2):
        before, after = checkpoints[step - 1], checkpoints[step]
        before_state, after_state = extractor_state(before), extractor_state(after)
        difference = parameter_difference(before_state, after_state, trainable)
        earlier_static, later_static = static_anchors(before), static_anchors(after)
        if earlier_static.keys() != later_static.keys() or any(
                not torch.equal(value, later_static[name]) for name, value in earlier_static.items()):
            raise AssertionError("Frozen static extractor/reference changed")
        before_updates = int(before["objective"]["critic_updates"])
        after_updates = int(after["objective"]["critic_updates"])
        update_count = after_updates - before_updates
        if after_updates != rows[step - 1]["critic_updates"] or update_count not in (0, 1):
            raise AssertionError("Critic update log/checkpoint mismatch")
        if bool(update_count) != rows[step - 1]["critic_updated"]:
            raise AssertionError("Critic update flag disagrees with checkpoint")
        if difference["unexpected_frozen_changes"]:
            raise AssertionError("Frozen dynamic extractor weights changed")
        real_delta = (features[step][0] - features[step - 1][0]).double().norm().item()
        fake_delta = (features[step][1] - features[step - 1][1]).double().norm().item()
        if update_count:
            if difference["changed_trainable_tensors"] == 0 or not difference["relative_l2_delta"] > 0:
                raise AssertionError("Critic optimizer stepped but no trainable MAE parameter changed")
            if not math.isfinite(rows[step - 1]["critic_grad_norm"]) or rows[step - 1]["critic_grad_norm"] <= 0:
                raise AssertionError("Missing positive finite D-step gradient")
            if max(real_delta, fake_delta) <= 10 * repeat_noise:
                raise AssertionError("MAE weights changed but fixed-probe features did not")
        elif difference["changed_trainable_tensors"] or real_delta > 10 * repeat_noise or fake_delta > 10 * repeat_noise:
            raise AssertionError("Dynamic extractor changed on a G-only step")
        reports.append({"step": step, "critic_updates_in_step": update_count,
                        "fixed_real_feature_l2_delta": real_delta,
                        "fixed_reconstruction_feature_l2_delta": fake_delta, **difference})

    expected = ([0, 1] if config["method"] == "advfd_reconstruction" else [1, 0])
    if [row["critic_updates_in_step"] for row in reports] != expected:
        raise AssertionError(f"Unexpected B/C preflight update sequence; expected {expected}")
    return {"status": "passed", "method": config["method"], "scope": manifest["scope"],
            "run": str(run_dir), "sample_ids": sample_ids, "probe_repeat_l2_noise": repeat_noise,
            "training_implementation_sha256": checkpoints[0]["implementation_sha256"],
            "checks": reports,
            "limitation": "Checkpoint intervals prove D-scheduled versus G-only changes, not within-step D/G isolation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--probe-count", type=int, default=8)
    args = parser.parse_args()
    if args.probe_count < 2:
        parser.error("--probe-count must be at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA audit requested without a GPU")
    try:
        result = audit_run(args.run, args.device, args.probe_count)
    except Exception as error:
        write_json(args.output, {"status": "failed", "run": args.run, "error": repr(error)})
        raise
    write_json(args.output, result)
    print(json.dumps({"dynamic_extractor_audit": result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
