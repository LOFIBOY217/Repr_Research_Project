#!/usr/bin/env python3
"""Generate a class-balanced ImageNet npz from one iMF model."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models
from models.denoiser_imf import convert_imf_checkpoint
from utils.builders import create_tokenizer
from utils.data_util import to_uint8_numpy


IMAGE_SIZE = 256
LATENT_CHANNELS = 4
TOKENIZER_PATCH_SIZE = 8
LATENT_SIZE = IMAGE_SIZE // TOKENIZER_PATCH_SIZE
NUM_SAMPLING_STEPS = 1
NOISE_SCALE = 1.0

MODEL_CONFIGS = {
    "imf-b": {
        "model": "iMF_B",
        "cfg": 8.0,
        "interval_min": 0.4,
        "interval_max": 0.65,
    },
    "imf-xl": {
        "model": "iMF_XL",
        "cfg": 8.0,
        "interval_min": 0.42,
        "interval_max": 0.62,
    },
}


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate a clean arr_0 npz from one iMF checkpoint")
    parser.add_argument("--preset", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--num-images", default=50_000, type=int)
    parser.add_argument("--num-classes", default=1_000, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cfg", default=None, type=float)
    parser.add_argument("--labels-npy", default=None)
    parser.add_argument("--labels-txt", default=None)
    return parser.parse_args()


def make_balanced_labels(num_images: int, num_classes: int) -> np.ndarray:
    base = num_images // num_classes
    remainder = num_images % num_classes
    labels = []
    for class_id in range(num_classes):
        count = base + (1 if class_id < remainder else 0)
        labels.extend([class_id] * count)
    return np.asarray(labels, dtype=np.int64)


def model_args(config: dict, checkpoint: str, batch_size: int,
               num_classes: int, cfg: float) -> SimpleNamespace:
    return SimpleNamespace(
        model=config["model"],
        load_from=checkpoint,
        tokenizer="sdvae",
        img_size=IMAGE_SIZE,
        token_channels=LATENT_CHANNELS,
        tokenizer_patch_size=TOKENIZER_PATCH_SIZE,
        patch_size=2,
        num_classes=num_classes,
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
        global_bsz=batch_size,
        cfg=cfg,
        interval_min=config["interval_min"],
        interval_max=config["interval_max"],
        num_sampling_steps=NUM_SAMPLING_STEPS,
        same_noise=False,
        enable_amp=True,
        amp_dtype=torch.bfloat16,
    )


def tokenizer_args() -> SimpleNamespace:
    return SimpleNamespace(tokenizer="sdvae")


def load_model(config: dict, checkpoint: Path, batch_size: int,
               num_classes: int, cfg: float) -> torch.nn.Module:
    args = model_args(config, str(checkpoint), batch_size, num_classes, cfg)
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

    if not checkpoint.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = convert_imf_checkpoint(state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"{config['model']}: loaded {checkpoint}: {msg}", flush=True)
    del ckpt, state_dict

    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def save_labels(labels: np.ndarray, labels_npy: Path, labels_txt: Path) -> None:
    labels_npy.parent.mkdir(parents=True, exist_ok=True)
    labels_txt.parent.mkdir(parents=True, exist_ok=True)
    np.save(labels_npy, labels)
    with labels_txt.open("w") as f:
        for idx, label in enumerate(labels):
            f.write(f"{idx}\t{int(label)}\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script inside a GPU allocation.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    config = dict(MODEL_CONFIGS[args.preset])
    cfg = config["cfg"] if args.cfg is None else args.cfg
    checkpoint = expand_path(args.checkpoint)
    output_npz = expand_path(args.output_npz)
    labels_npy = expand_path(args.labels_npy) if args.labels_npy else output_npz.with_suffix(".labels.npy")
    labels_txt = expand_path(args.labels_txt) if args.labels_txt else output_npz.with_suffix(".labels.txt")
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    labels = make_balanced_labels(args.num_images, args.num_classes)
    if len(labels) != args.num_images:
        raise RuntimeError(f"built {len(labels)} labels for {args.num_images} images")
    save_labels(labels, labels_npy, labels_txt)

    model = load_model(config, checkpoint, args.batch_size, args.num_classes, cfg)
    gen_args = model_args(config, str(checkpoint), args.batch_size, args.num_classes, cfg)
    latents_out = torch.empty(
        args.num_images, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
    )

    print(f"preset={args.preset} model={config['model']} cfg={cfg}", flush=True)
    print(f"output_npz={output_npz}", flush=True)
    print(f"num_images={args.num_images} num_classes={args.num_classes} batch_size={args.batch_size}", flush=True)
    print(f"latent_shape=({LATENT_CHANNELS}, {LATENT_SIZE}, {LATENT_SIZE})", flush=True)
    print(f"labels_per_class_base={args.num_images // args.num_classes}", flush=True)

    for start in range(0, args.num_images, args.batch_size):
        end = min(start + args.batch_size, args.num_images)
        batch_labels = torch.from_numpy(labels[start:end]).to("cuda", dtype=torch.long)
        z_t = (
            torch.randn(
                end - start, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE,
                device="cuda", dtype=torch.bfloat16,
            )
            * NOISE_SCALE
        )
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            latents = model.generate(
                n_samples=batch_labels.shape[0],
                labels=batch_labels,
                cfg=cfg,
                args=gen_args,
                verbose=False,
                z_t=z_t,
            )
        latents_out[start:end] = latents.detach().to("cpu")

        del batch_labels, z_t, latents
        if (end % 1000 == 0) or (end == args.num_images):
            print(f"generated {end}/{args.num_images} images", flush=True)
            torch.cuda.empty_cache()
            gc.collect()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    tokenizer = create_tokenizer(tokenizer_args()).eval().requires_grad_(False)
    images_out = np.empty((args.num_images, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for start in range(0, args.num_images, args.batch_size):
        end = min(start + args.batch_size, args.num_images)
        latents = latents_out[start:end].to("cuda", non_blocking=True)
        images = tokenizer.detokenize(latents)
        images_out[start:end] = to_uint8_numpy(images)
        del latents, images
        if (end % 1000 == 0) or (end == args.num_images):
            print(f"decoded {end}/{args.num_images} images", flush=True)
            torch.cuda.empty_cache()
            gc.collect()

    np.savez(output_npz, images_out)
    print(f"saved npz to {output_npz}", flush=True)
    print(f"saved labels to {labels_npy} and {labels_txt}", flush=True)
    print(f"arr_0 shape={images_out.shape} dtype={images_out.dtype} min={images_out.min()} max={images_out.max()}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
