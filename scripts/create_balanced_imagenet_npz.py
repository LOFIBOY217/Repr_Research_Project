#!/usr/bin/env python3
"""Create a class-balanced ImageNet image npz for evaluator.py."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_util import center_crop_arr


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_DATA_ROOT = "/datashare/imagenet/ILSVRC2012"
DEFAULT_OUTPUT_NPZ = (
    "~/projects/def-kdhkdh/jiaqi217/data/eval_refs/"
    "imagenet_val_50k_256_balanced.npz"
)


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample ImageNet uniformly by class, center-crop to 256, and save arr_0 npz.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help=f"ImageNet root containing train/val. Default: {DEFAULT_DATA_ROOT}")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output-npz", default=DEFAULT_OUTPUT_NPZ,
                        help=f"Output npz path. Default: {DEFAULT_OUTPUT_NPZ}")
    parser.add_argument("--num-images", default=50_000, type=int)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--no-compress", action="store_true",
                        help="Use np.savez instead of np.savez_compressed.")
    parser.add_argument("--manifest", default=None,
                        help="Optional text file with one 'class path' row per selected image.")
    return parser.parse_args()


def find_class_dirs(split_dir: Path) -> list[Path]:
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found under {split_dir}")
    return class_dirs


def find_images(class_dir: Path) -> list[Path]:
    return sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def allocate_counts(num_images: int, num_classes: int) -> list[int]:
    base = num_images // num_classes
    remainder = num_images % num_classes
    return [base + (1 if idx < remainder else 0) for idx in range(num_classes)]


def select_images(class_dirs: list[Path], num_images: int, seed: int) -> list[tuple[str, Path]]:
    rng = np.random.default_rng(seed)
    counts = allocate_counts(num_images, len(class_dirs))
    selected: list[tuple[str, Path]] = []

    for class_dir, count in zip(class_dirs, counts):
        images = find_images(class_dir)
        if len(images) < count:
            raise ValueError(
                f"Class {class_dir.name} only has {len(images)} images, need {count}"
            )
        if count == len(images):
            chosen = images
        else:
            indices = np.sort(rng.choice(len(images), size=count, replace=False))
            chosen = [images[i] for i in indices]
        selected.extend((class_dir.name, path) for path in chosen)

    return selected


def load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    cropped = center_crop_arr(image, image_size)
    array = np.asarray(cropped, dtype=np.uint8)
    if array.shape != (image_size, image_size, 3):
        raise ValueError(f"{path} produced unexpected shape {array.shape}")
    return array


def main() -> None:
    args = parse_args()
    data_root = expand_path(args.data_root)
    split_dir = data_root / args.split
    output_npz = expand_path(args.output_npz)
    manifest = expand_path(args.manifest) if args.manifest else output_npz.with_suffix(".txt")

    class_dirs = find_class_dirs(split_dir)
    selected = select_images(class_dirs, args.num_images, args.seed)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    images = np.empty(
        (len(selected), args.image_size, args.image_size, 3),
        dtype=np.uint8,
    )

    with manifest.open("w") as f:
        for idx, (class_name, path) in enumerate(selected):
            images[idx] = load_image(path, args.image_size)
            f.write(f"{class_name}\t{path}\n")
            if (idx + 1) % 1000 == 0 or idx + 1 == len(selected):
                print(f"processed {idx + 1}/{len(selected)} images", flush=True)

    if args.no_compress:
        np.savez(output_npz, images)
    else:
        np.savez_compressed(output_npz, images)

    print(f"saved npz to {output_npz}", flush=True)
    print(f"saved manifest to {manifest}", flush=True)
    print(f"arr_0 shape={images.shape} dtype={images.dtype} min={images.min()} max={images.max()}", flush=True)
    print(f"classes={len(class_dirs)} images_per_class={args.num_images // len(class_dirs)}", flush=True)


if __name__ == "__main__":
    main()
