import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from recon_fd.data import sequential_loader
from recon_fd.provenance import write_json, state_fingerprint, implementation_fingerprint
from recon_fd.objectives.statistics import RunningMoments
from recon_fd.objectives.frechet import frechet_distance
from recon_fd.representations import build_representation
from .reference import get_reference
from .progress import log_progress, should_log


def quantize(images):
    return (images.detach().clamp(0, 1) * 255 + 0.5).floor().to(torch.uint8)


def psnr_per_image(real, reconstruction):
    mse = (real - reconstruction).square().flatten(1).mean(1)
    # Exact identity is +inf mathematically; cap at 120 dB and label this policy.
    return -10 * mse.clamp_min(1e-12).log10()


def verify_manifest(directory, expected_ids, expected_identity):
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        raise ValueError("Reconstruction export is incomplete")
    result = json.loads(marker.read_text())
    if result["identity"] != expected_identity or result["sample_ids"] != list(expected_ids):
        raise ValueError("Export dataset, checkpoint, or sample order mismatch")
    names = [f"{i:06d}.png" for i in range(len(expected_ids))]
    if sorted(p.name for p in directory.glob("*.png")) != names or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Missing, extra, or duplicate reconstruction samples")
    return result


@torch.no_grad()
def export_reconstructions(model, dataset, directory, config, device):
    directory = Path(directory)
    identity = {"dataset": dataset.identity, "model_sha256": state_fingerprint(model), "quantization": "round_half_up_uint8_v1"}
    if directory.exists() and any(directory.iterdir()):
        return verify_manifest(directory, dataset.ids, identity)
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    offset = 0
    for batch in sequential_loader(dataset, config["evaluation"]["batch_size"], config["data"]["workers"]):
        images = model(batch["image"].to(device))
        if not torch.isfinite(images).all() or images.shape != batch["image"].shape:
            raise ValueError("Invalid reconstruction during export")
        arrays = quantize(images).permute(0, 2, 3, 1).cpu().numpy()
        previous = offset
        for array in arrays:
            Image.fromarray(array).save(directory / f"{offset:06d}.png")
            offset += 1
        if should_log(offset, previous, len(dataset)):
            log_progress("export", offset, len(dataset))
    if offset != len(dataset):
        raise ValueError("Export count mismatch")
    result = {"identity": identity, "sample_ids": dataset.ids, "count": offset}
    write_json(directory / "complete.json", result)
    return verify_manifest(directory, dataset.ids, identity)


def reconstructed_batch(directory, offset, n):
    values = []
    for i in range(offset, offset + n):
        with Image.open(Path(directory) / f"{i:06d}.png") as image:
            values.append(torch.from_numpy(np.array(image.convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255)
    return torch.stack(values)


@torch.no_grad()
def evaluate_export(dataset, directory, output, config, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {"schema": 1, "engineering_only": config["evaluation"]["engineering_only"],
               "num_samples": len(dataset), "dataset": dataset.identity,
               "image_protocol": "uint8_png_recon_vs_float01_real_center_crop_v1",
               "implementation_sha256": implementation_fingerprint(),
               "training_representation_names": [s["name"] for s in config["static"]["representations"]],
               "fd": {}, "representations": {}, "paired": {}}
    loader = sequential_loader(dataset, config["evaluation"]["batch_size"], config["data"]["workers"])
    for spec in config["evaluation"]["representations"]:
        extractor = build_representation(spec).to(device)
        reference, identity = get_reference(extractor, dataset, config["static"]["reference_cache"],
                                            config["evaluation"]["batch_size"], device, config["data"]["workers"])
        accumulator, offset = RunningMoments(), 0
        for batch in loader:
            n = len(batch["id"])
            reconstructions = reconstructed_batch(directory, offset, n).to(device)
            accumulator.update(extractor(reconstructions))
            previous = offset
            offset += n
            if should_log(offset, previous, len(dataset)):
                log_progress(f"features {spec['name']}", offset, len(dataset))
        current = accumulator.moments()
        results["fd"][spec["name"]] = float(frechet_distance(reference.mean, reference.cov, current.mean, current.cov))
        results["representations"][spec["name"]] = identity
        del extractor, accumulator, reference, current
        if device.type == "cuda":
            torch.cuda.empty_cache()
    metrics = config["evaluation"]["paired_metrics"]
    if not set(metrics) <= {"psnr", "ssim", "lpips"}:
        raise ValueError("Unknown paired metric; no silent skipping")
    lpips_model = None
    if "lpips" in metrics:
        import lpips
        lpips_model = lpips.LPIPS(net="alex").eval().requires_grad_(False).to(device)
    sums, count = dict.fromkeys(metrics, 0.0), 0
    with (output / "per_image.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", *metrics, "pixel_mse", "residual_laplacian_energy"])
        writer.writeheader()
        for batch in loader:
            real = batch["image"].to(device)
            recon = reconstructed_batch(directory, count, len(real)).to(device)
            values = {}
            if "psnr" in metrics:
                values["psnr"] = psnr_per_image(real, recon).cpu().tolist()
            if "ssim" in metrics:
                from skimage.metrics import structural_similarity
                values["ssim"] = [float(structural_similarity(x.permute(1, 2, 0).cpu().numpy(),
                                     y.permute(1, 2, 0).cpu().numpy(), data_range=1.0, channel_axis=2)) for x, y in zip(real, recon)]
            if lpips_model is not None:
                values["lpips"] = lpips_model(real * 2 - 1, recon * 2 - 1).flatten().cpu().tolist()
            residual = real - recon
            mse = residual.square().flatten(1).mean(1).cpu().tolist()
            laplacian = (-4 * residual[:, :, 1:-1, 1:-1] + residual[:, :, :-2, 1:-1] + residual[:, :, 2:, 1:-1]
                         + residual[:, :, 1:-1, :-2] + residual[:, :, 1:-1, 2:])
            energy = laplacian.square().flatten(1).mean(1).cpu().tolist()
            for i, sample_id in enumerate(batch["id"]):
                row = {"sample_id": sample_id, "pixel_mse": mse[i], "residual_laplacian_energy": energy[i]}
                for name in metrics:
                    row[name] = values[name][i]
                    sums[name] += values[name][i]
                writer.writerow(row)
            count += len(real)
            if should_log(count, count - len(real), len(dataset)):
                log_progress("paired metrics", count, len(dataset))
    if count != len(dataset):
        raise ValueError("Paired evaluation count mismatch")
    results["paired"] = {k: v / count for k, v in sums.items()}
    results["psnr_zero_mse_policy"] = "mse floor 1e-12 (120 dB maximum)"
    results["artifact_proxy_warning"] = "Residual energy is a screening proxy, not proof of visual artifacts."
    write_json(output / "metrics.json", results)
    return results
