#!/usr/bin/env python3
"""Generate RGB images from iMF checkpoints."""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from main_fd import get_args_parser as get_training_args_parser
from utils.builders import create_generation_model, create_tokenizer
from utils.checkpoint_util import ckpt_resume
from utils.data_util import to_uint8_numpy


PRESETS = {
    "imf": {
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

TOKENIZER_SPECS = {
    "sdvae": {"channels": 4, "patch_size": 8},
}

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

logger = logging.getLogger("FD_loss")


def option_was_set(argv: list[str], *names: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv for name in names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Generate sample images from iMF checkpoints",
        parents=[get_training_args_parser()],
    )
    parser.add_argument("--preset", default="imf", choices=sorted(PRESETS))
    parser.add_argument("--checkpoint", "--load-from", dest="load_from", type=str)
    parser.add_argument("--compare-preset", choices=sorted(PRESETS),
                        help="Optional second preset for side-by-side comparison.")
    parser.add_argument("--compare-checkpoint", type=str,
                        help="Checkpoint for --compare-preset.")
    parser.add_argument("--output-dir", dest="sample_output_dir", type=Path, required=True)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-images", dest="num_images", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-sampling-steps", dest="num_sampling_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--interval-min", dest="interval_min", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--interval-max", dest="interval_max", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--noise-scale", dest="noise_scale", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--img-size", dest="img_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--patch-size", dest="patch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--token-channels", dest="token_channels", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--tokenizer-patch-size", dest="tokenizer_patch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--disable-v-head", dest="disable_v_head", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--class-labels", type=int, nargs="+", default=None,
                        help="Optional ImageNet class ids. If omitted, labels are sampled uniformly.")
    parser.add_argument("--ema-label", default="online",
                        help="Use 'online' weights or one EMA label from the checkpoint.")
    parser.add_argument("--decode-bsz", default=None, type=int,
                        help="Override VAE decode chunk size.")
    parser.add_argument("--save-latents", action="store_true",
                        help="Save the initial noise and final normalized latents as .pt files.")
    parser.add_argument("--save-grid", action="store_true",
                        help="Also save grid.png with all generated images.")
    parser.add_argument("--filename-prefix", default="sample")

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.load_from is None:
        parser.error("one of --checkpoint, --load-from, or --load_from is required")
    if (args.compare_preset is None) != (args.compare_checkpoint is None):
        parser.error("--compare-preset and --compare-checkpoint must be used together")
    preset = PRESETS[args.preset]

    if not option_was_set(argv, "--model"):
        args.model = preset["model"]
    if not option_was_set(argv, "--cfg"):
        args.cfg = preset["cfg"]
    if not option_was_set(argv, "--interval_min", "--interval-min"):
        args.interval_min = preset["interval_min"]
    if not option_was_set(argv, "--interval_max", "--interval-max"):
        args.interval_max = preset["interval_max"]
    if not option_was_set(argv, "--tokenizer"):
        args.tokenizer = "sdvae"
    if not option_was_set(argv, "--patch_size", "--patch-size"):
        args.patch_size = 2
    if not option_was_set(argv, "--disable_v_head", "--disable-v-head"):
        args.disable_v_head = True
    if args.tokenizer in TOKENIZER_SPECS:
        spec = TOKENIZER_SPECS[args.tokenizer]
        if not option_was_set(argv, "--token_channels", "--token-channels"):
            args.token_channels = spec["channels"]
        if not option_was_set(argv, "--tokenizer_patch_size", "--tokenizer-patch-size"):
            args.tokenizer_patch_size = spec["patch_size"]

    args.auto_resume = False
    args.resume_from = None
    args.enable_amp = args.dtype != "fp32"
    args.amp_dtype = DTYPE_MAP[args.dtype]
    args.global_bsz = args.batch_size
    args.user_overrode_cfg = option_was_set(argv, "--cfg")
    args.user_overrode_interval_min = option_was_set(argv, "--interval_min", "--interval-min")
    args.user_overrode_interval_max = option_was_set(argv, "--interval_max", "--interval-max")
    return args


def args_for_preset(base_args: argparse.Namespace, preset_name: str,
                    checkpoint: str) -> argparse.Namespace:
    model_args = copy.deepcopy(base_args)
    preset = PRESETS[preset_name]
    model_args.preset = preset_name
    model_args.model = preset["model"]
    model_args.cfg = base_args.cfg if base_args.user_overrode_cfg else preset["cfg"]
    model_args.interval_min = (
        base_args.interval_min if base_args.user_overrode_interval_min else preset["interval_min"]
    )
    model_args.interval_max = (
        base_args.interval_max if base_args.user_overrode_interval_max else preset["interval_max"]
    )
    model_args.load_from = checkpoint
    return model_args


def make_labels(args: argparse.Namespace, batch_size: int) -> torch.Tensor:
    if args.class_labels is None:
        return torch.randint(0, args.num_classes, (batch_size,), device="cuda")

    labels = torch.tensor(args.class_labels, dtype=torch.long, device="cuda")
    if torch.any((labels < 0) | (labels >= args.num_classes)):
        raise ValueError(f"class labels must be in [0, {args.num_classes - 1}]")
    repeats = math.ceil(batch_size / labels.numel())
    return labels.repeat(repeats)[:batch_size]


def save_images(images: torch.Tensor, labels: torch.Tensor, output_dir: Path,
                start_index: int, filename_prefix: str) -> None:
    arrays = to_uint8_numpy(images)
    labels_cpu = labels.detach().cpu().tolist()
    for offset, (array, label) in enumerate(zip(arrays, labels_cpu)):
        path = output_dir / f"{filename_prefix}_{start_index + offset:06d}_class{label:04d}.png"
        Image.fromarray(array).save(path)


def save_compare_images(left_images: torch.Tensor, right_images: torch.Tensor,
                        labels: torch.Tensor, output_dir: Path, start_index: int,
                        left_name: str, right_name: str,
                        filename_prefix: str) -> None:
    left_np = to_uint8_numpy(left_images)
    right_np = to_uint8_numpy(right_images)
    labels_cpu = labels.detach().cpu().tolist()
    for offset, (left, right, label) in enumerate(zip(left_np, right_np, labels_cpu)):
        h, w = left.shape[:2]
        comparison = Image.new("RGB", (w * 2, h))
        comparison.paste(Image.fromarray(left), (0, 0))
        comparison.paste(Image.fromarray(right), (w, 0))

        index = start_index + offset
        stem = f"{filename_prefix}_{index:06d}_class{label:04d}"
        Image.fromarray(left).save(output_dir / f"{stem}_{left_name}.png")
        Image.fromarray(right).save(output_dir / f"{stem}_{right_name}.png")
        comparison.save(output_dir / f"{stem}_compare.png")


def save_grid(images: torch.Tensor, output_dir: Path) -> None:
    arrays = to_uint8_numpy(images)
    if len(arrays) == 0:
        return
    h, w = arrays[0].shape[:2]
    cols = math.ceil(math.sqrt(len(arrays)))
    rows = math.ceil(len(arrays) / cols)
    grid = Image.new("RGB", (cols * w, rows * h))
    for idx, array in enumerate(arrays):
        grid.paste(Image.fromarray(array), ((idx % cols) * w, (idx // cols) * h))
    grid.save(output_dir / "grid.png")


def generate_batches(args: argparse.Namespace, latent_shape: tuple[int, int, int]) -> list[dict]:
    batches = []
    generated = 0
    while generated < args.num_images:
        batch_size = min(args.batch_size, args.num_images - generated)
        labels = make_labels(args, batch_size)
        z_t = torch.randn(batch_size, *latent_shape, device="cuda") * args.noise_scale
        batches.append({
            "start_index": generated,
            "labels": labels.detach().cpu(),
            "z_t": z_t.detach().cpu(),
        })
        generated += batch_size
    return batches


def generate_from_batches(args: argparse.Namespace, tokenizer: torch.nn.Module,
                          batches: list[dict]) -> list[torch.Tensor]:
    model, ema_model = create_generation_model(args)
    ckpt_resume(args, model, optimizer=None, model_ema=ema_model)
    model.eval()

    images_cpu = []
    ema_context = (
        ema_model.swap(model, label=args.ema_label)
        if args.ema_label != "online" else nullcontext()
    )
    with ema_context:
        for batch in batches:
            labels = batch["labels"].to("cuda", non_blocking=True)
            z_t = batch["z_t"].to("cuda", non_blocking=True)
            with torch.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
                z_final = model.generate(
                    n_samples=labels.shape[0],
                    labels=labels,
                    cfg=args.cfg,
                    args=args,
                    verbose=args.num_sampling_steps > 2,
                    z_t=z_t,
                )
                images = tokenizer.detokenize(z_final, decode_bsz=args.decode_bsz)
            images_cpu.append(images.detach().cpu())
            print(f"{args.preset}: generated {batch['start_index'] + labels.shape[0]}/{args.num_images}",
                  flush=True)

    del model, ema_model
    torch.cuda.empty_cache()
    gc.collect()
    return images_cpu


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script inside a GPU allocation.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)

    output_dir = args.sample_output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = create_tokenizer(args)
    if args.tokenizer is None:
        raise ValueError("iMF generation needs a tokenizer so normalized latents can decode to RGB.")

    latent_h = args.img_size // args.tokenizer_patch_size
    latent_shape = (args.token_channels, latent_h, latent_h)
    print(f"latent_shape={latent_shape} cfg={args.cfg} noise_scale={args.noise_scale}")
    print(f"output_dir={output_dir}")

    batches = generate_batches(args, latent_shape)
    if args.save_latents:
        torch.save(
            {
                "batches": batches,
                "latent_shape": latent_shape,
                "noise_scale": args.noise_scale,
                "num_classes": args.num_classes,
            },
            output_dir / f"{args.filename_prefix}_inputs.pt",
        )

    if args.compare_preset is None:
        print(f"model={args.model} checkpoint={args.load_from}")
        images_by_batch = generate_from_batches(args, tokenizer, batches)
        for images, batch in zip(images_by_batch, batches):
            save_images(images, batch["labels"], output_dir, batch["start_index"], args.filename_prefix)
        if args.save_grid:
            save_grid(torch.cat(images_by_batch, dim=0), output_dir)
    else:
        left_args = args_for_preset(args, args.preset, args.load_from)
        right_args = args_for_preset(args, args.compare_preset, args.compare_checkpoint)
        print(f"left={left_args.model} checkpoint={left_args.load_from}")
        print(f"right={right_args.model} checkpoint={right_args.load_from}")

        left_images = generate_from_batches(left_args, tokenizer, batches)
        right_images = generate_from_batches(right_args, tokenizer, batches)
        for left, right, batch in zip(left_images, right_images, batches):
            save_compare_images(
                left, right, batch["labels"], output_dir, batch["start_index"],
                left_args.preset, right_args.preset, args.filename_prefix,
            )
        if args.save_grid:
            compare_rows = []
            for left, right in zip(left_images, right_images):
                pairs = torch.stack([left, right], dim=1).flatten(0, 1)
                compare_rows.append(pairs)
            save_grid(torch.cat(compare_rows, dim=0), output_dir)
    print(f"saved generated RGB images to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
