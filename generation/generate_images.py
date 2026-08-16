#!/usr/bin/env python3
"""Generate side-by-side iMF vs iMF-XL comparison images."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

import models
from models.denoiser_imf import convert_imf_checkpoint
from utils.builders import create_tokenizer
from utils.data_util import to_uint8_numpy


IMAGE_SIZE = 256
LATENT_CHANNELS = 4
TOKENIZER_PATCH_SIZE = 8
LATENT_SIZE = IMAGE_SIZE // TOKENIZER_PATCH_SIZE
NUM_CLASSES = 1000
NOISE_SCALE = 1.0
NUM_SAMPLING_STEPS = 1

IMF_CONFIG = {
    "preset": "imf",
    "model": "iMF_B",
    "cfg": 8.0,
    "interval_min": 0.4,
    "interval_max": 0.65,
}

IMF_XL_CONFIG = {
    "preset": "imf-xl",
    "model": "iMF_XL",
    "cfg": 8.0,
    "interval_min": 0.42,
    "interval_max": 0.62,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate iMF vs iMF-XL compare PNGs")
    parser.add_argument("--imf-checkpoint", required=True)
    parser.add_argument("--imf-xl-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def model_args(config: dict, checkpoint: str, batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=config["model"],
        load_from=checkpoint,
        auto_resume=False,
        resume_from=None,
        ckpt_dir=None,
        tokenizer="sdvae",
        img_size=IMAGE_SIZE,
        token_channels=LATENT_CHANNELS,
        tokenizer_patch_size=TOKENIZER_PATCH_SIZE,
        patch_size=2,
        num_classes=NUM_CLASSES,
        label_drop_prob=0.1,
        P_mean=0.8,
        P_std=0.8,
        ratio_r_neq_t=0.5,
        cfg_beta=1.0,
        cfg_omega_max=7.0,
        aux_head_depth=8,
        class_tokens=8,
        time_tokens=4,
        guidance_tokens=4,
        interval_tokens=2,
        rope_2d=False,
        learned_pe=False,
        disable_v_head=True,
        ema_type="edm",
        ema_rates=[0.9999, 0.9996],
        ema_halflife_kimg=[250, 500, 1000, 2000],
        global_bsz=batch_size,
        cfg=config["cfg"],
        interval_min=config["interval_min"],
        interval_max=config["interval_max"],
        num_sampling_steps=NUM_SAMPLING_STEPS,
        same_noise=False,
        enable_amp=True,
        amp_dtype=torch.bfloat16,
    )


def tokenizer_args() -> SimpleNamespace:
    return SimpleNamespace(tokenizer="sdvae")


def make_inputs(num_images: int, batch_size: int) -> list[dict]:
    batches = []
    generated = 0
    while generated < num_images:
        bsz = min(batch_size, num_images - generated)
        labels = torch.randint(0, NUM_CLASSES, (bsz,), device="cuda")
        z_t = torch.randn(bsz, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, device="cuda") * NOISE_SCALE
        batches.append({
            "start_index": generated,
            "labels": labels.detach().cpu(),
            "z_t": z_t.detach().cpu(),
        })
        generated += bsz
    return batches


def load_model(config: dict, checkpoint: str, batch_size: int) -> torch.nn.Module:
    args = model_args(config, checkpoint, batch_size)
    model = models.iMFDenoiser_models[args.model](
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.token_channels,
        tokenizer_patch_size=args.tokenizer_patch_size,
        num_classes=args.num_classes,
        label_drop_prob=args.label_drop_prob,
        P_mean=args.P_mean,
        P_std=args.P_std,
        ratio_r_neq_t=args.ratio_r_neq_t,
        cfg_beta=args.cfg_beta,
        cfg_omega_max=args.cfg_omega_max,
        aux_head_depth=args.aux_head_depth,
        class_tokens=args.class_tokens,
        time_tokens=args.time_tokens,
        guidance_tokens=args.guidance_tokens,
        interval_tokens=args.interval_tokens,
        rope_2d=args.rope_2d,
        learned_pe=args.learned_pe,
        disable_v_head=args.disable_v_head,
    )

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = convert_imf_checkpoint(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"{config['preset']}: loaded {checkpoint}: {msg}", flush=True)
    del ckpt, state_dict

    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def generate_model_latents(config: dict, checkpoint: str,
                           batches: list[dict], batch_size: int) -> list[torch.Tensor]:
    args = model_args(config, checkpoint, batch_size)
    model = load_model(config, checkpoint, batch_size)

    latents_by_batch = []
    for batch in batches:
        labels = batch["labels"].to("cuda", non_blocking=True)
        z_t = batch["z_t"].to("cuda", dtype=torch.bfloat16, non_blocking=True)
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            latents = model.generate(
                n_samples=labels.shape[0],
                labels=labels,
                cfg=args.cfg,
                args=args,
                verbose=False,
                z_t=z_t,
            )
        latents_by_batch.append(latents.detach().cpu())
        print(f"{config['preset']}: generated {batch['start_index'] + labels.shape[0]}", flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return latents_by_batch


def save_compare_pngs(imf_latents: list[torch.Tensor], imf_xl_latents: list[torch.Tensor],
                      batches: list[dict], output_dir: Path) -> None:
    tokenizer = create_tokenizer(tokenizer_args())
    for left_latents, right_latents, batch in zip(imf_latents, imf_xl_latents, batches):
        left_images = tokenizer.detokenize(left_latents.to("cuda", non_blocking=True))
        right_images = tokenizer.detokenize(right_latents.to("cuda", non_blocking=True))
        left_np = to_uint8_numpy(left_images)
        right_np = to_uint8_numpy(right_images)
        labels = batch["labels"].tolist()
        for offset, (left, right, label) in enumerate(zip(left_np, right_np, labels)):
            h, w = left.shape[:2]
            comparison = Image.new("RGB", (w * 2, h))
            comparison.paste(Image.fromarray(left), (0, 0))
            comparison.paste(Image.fromarray(right), (w, 0))
            index = batch["start_index"] + offset
            comparison.save(output_dir / f"compare_{index:06d}_class{label:04d}.png")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script inside a GPU allocation.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = make_inputs(args.num_images, args.batch_size)

    print(f"output_dir={output_dir}")
    print(f"num_images={args.num_images} batch_size={args.batch_size}")
    print(f"latent_shape=({LATENT_CHANNELS}, {LATENT_SIZE}, {LATENT_SIZE})")

    imf_latents = generate_model_latents(
        IMF_CONFIG, args.imf_checkpoint, batches, args.batch_size,
    )
    imf_xl_latents = generate_model_latents(
        IMF_XL_CONFIG, args.imf_xl_checkpoint, batches, args.batch_size,
    )
    save_compare_pngs(imf_latents, imf_xl_latents, batches, output_dir)
    print(f"saved compare PNGs to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
