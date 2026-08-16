#!/usr/bin/env python3
"""Reconstruct an image npz through the repo VAE and save a new npz."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.autoencoder import DiffusersAutoencoderKL
from utils.data_util import to_uint8_numpy


DEFAULT_INPUT_NPZ = (
    "~/projects/def-kdhkdh/jiaqi217/data/eval_refs/"
    "VIRTUAL_imagenet256_labeled.npz"
)
DEFAULT_OUTPUT_NPZ = (
    "~/projects/def-kdhkdh/jiaqi217/data/eval_refs/"
    "VIRTUAL_imagenet256_labeled_reconstructed_sdvae.npz"
)


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct arr_0 from an ImageNet-style npz through the VAE.",
    )
    parser.add_argument("--input-npz", default=DEFAULT_INPUT_NPZ,
                        help=f"Input npz with RGB images in arr_0. Default: {DEFAULT_INPUT_NPZ}")
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ,
                        help=f"Output npz with reconstructed RGB images in arr_0. Default: {DEFAULT_OUTPUT_NPZ}")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--max-images", default=None, type=int,
                        help="Only reconstruct the first N images for a smoke test.")
    return parser.parse_args()


def validate_images(images: np.ndarray) -> None:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"arr_0 must have shape [N, H, W, 3], got {images.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"arr_0 must be uint8 RGB images, got {images.dtype}")


def to_nchw_float(batch: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    return tensor.to(dtype=torch.float32).div_(255.0)


def main() -> None:
    args = parse_args()
    input_npz = expand_path(args.input_npz)
    output_npz = expand_path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script inside a GPU allocation.")

    torch.backends.cuda.matmul.allow_tf32 = True

    data = np.load(input_npz)
    if "arr_0" not in data.files:
        raise ValueError(f"{input_npz} does not contain arr_0")

    images = data["arr_0"]
    validate_images(images)

    total = images.shape[0] if args.max_images is None else min(args.max_images, images.shape[0])
    reconstructions = np.empty((total, *images.shape[1:]), dtype=np.uint8)

    tokenizer = DiffusersAutoencoderKL(name="sdvae").eval().requires_grad_(False)

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = to_nchw_float(images[start:end]).cuda(non_blocking=True)
        recon = tokenizer.reconstruct(batch)
        reconstructions[start:end] = to_uint8_numpy(recon)
        print(f"reconstructed {end}/{total} images", flush=True)

    np.savez(output_npz, reconstructions)

    print(f"saved reconstructed npz to {output_npz}", flush=True)
    print(f"arr_0 shape={reconstructions.shape} dtype={reconstructions.dtype}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
