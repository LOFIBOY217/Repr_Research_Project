#!/usr/bin/env python3
"""Reconstruct image folders through the repository tokenizer/VAE."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.autoencoder import DiffusersAutoencoderKL
from utils.data_util import center_crop_arr, to_uint8_numpy


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class ReconstructionDataset(Dataset):
    def __init__(self, input_dir: Path, image_size: int, recursive: bool):
        self.input_dir = input_dir
        pattern = "**/*" if recursive else "*"
        self.paths = sorted(
            p for p in input_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {input_dir}")

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), str(path.relative_to(self.input_dir))


def save_batch(images: torch.Tensor, rel_paths: list[str], output_dir: Path, suffix: str) -> None:
    arrays = to_uint8_numpy(images)
    for array, rel_path in zip(arrays, rel_paths):
        rel = Path(rel_path)
        out_path = output_dir / rel.with_name(f"{rel.stem}{suffix}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(out_path)


def save_comparisons(originals: torch.Tensor, reconstructions: torch.Tensor,
                     rel_paths: list[str], output_dir: Path) -> None:
    originals_np = to_uint8_numpy(originals)
    recon_np = to_uint8_numpy(reconstructions)
    for original, recon, rel_path in zip(originals_np, recon_np, rel_paths):
        rel = Path(rel_path)
        comparison = Image.new("RGB", (original.shape[1] * 2, original.shape[0]))
        comparison.paste(Image.fromarray(original), (0, 0))
        comparison.paste(Image.fromarray(recon), (original.shape[1], 0))
        out_path = output_dir / rel.with_name(f"{rel.stem}_compare.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center-crop images, encode to normalized VAE latents, decode, and save reconstructions.",
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--img-size", default=256, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--tokenizer", default="sdvae", choices=["sdvae"])
    parser.add_argument("--suffix", default="_recon")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--sample-posterior", action="store_true",
                        help="Sample from the VAE posterior instead of using deterministic mode().")
    parser.add_argument("--decode-bsz", default=None, type=int,
                        help="Override VAE decode chunk size.")
    parser.add_argument("--max-images", default=None, type=int,
                        help="Only reconstruct the first N images.")
    parser.add_argument("--save-inputs", action="store_true",
                        help="Save the center-cropped input images next to reconstructions.")
    parser.add_argument("--save-comparisons", action="store_true",
                        help="Save side-by-side input/reconstruction images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script inside a GPU allocation.")

    torch.backends.cuda.matmul.allow_tf32 = True
    dataset = ReconstructionDataset(input_dir, args.img_size, args.recursive)
    if args.max_images is not None:
        dataset.paths = dataset.paths[:args.max_images]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    tokenizer = DiffusersAutoencoderKL(name=args.tokenizer).eval().requires_grad_(False)

    total = 0
    for images, rel_paths in loader:
        images = images.cuda(non_blocking=True)
        recon = tokenizer.reconstruct(
            images,
            sample=args.sample_posterior,
            decode_bsz=args.decode_bsz,
        )
        if args.save_inputs:
            save_batch(images, list(rel_paths), output_dir, "_input")
        save_batch(recon, list(rel_paths), output_dir, args.suffix)
        if args.save_comparisons:
            save_comparisons(images, recon, list(rel_paths), output_dir)
        total += images.shape[0]
        print(f"reconstructed {total}/{len(dataset)} images", flush=True)

    print(f"saved reconstructions to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
