"""Export paired ImageNet reconstruction examples from trusted B/C checkpoints.

This is a qualitative preview, not a 50k FD evaluation. Both checkpoints run
on the same CPU and the same center-cropped input tensors.
"""
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from recon_fd.data import center_crop
from recon_fd.engine.checkpoint import read_checkpoint
from recon_fd.evaluation.runner import quantize
from recon_fd.provenance import state_fingerprint
from recon_fd.tokenizers import build_tokenizer


# Names are the official ILSVRC2012 synset labels.
CLASSES = (
    ("n02099601", "golden retriever"),
    ("n02124075", "Egyptian cat"),
    ("n03417042", "garbage truck"),
    ("n07753592", "banana"),
    ("n02870880", "bookcase"),
    ("n02676566", "acoustic guitar"),
)


def select_examples(root, per_class, seed):
    root = Path(root)
    rng = random.Random(seed)
    examples = []
    for synset, label in CLASSES:
        paths = sorted(path for path in (root / synset).iterdir()
                       if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if len(paths) < per_class:
            raise ValueError(f"{synset} has only {len(paths)} images")
        for index, path in enumerate(sorted(rng.sample(paths, per_class))):
            examples.append({"synset": synset, "label": label, "index": index,
                             "sample_id": path.relative_to(root).as_posix(), "path": path})
    return examples


def save_tensor(image, path):
    pixels = quantize(image.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
    Image.fromarray(pixels).save(path)


def load_model(checkpoint, method, step):
    state = read_checkpoint(checkpoint)
    if state["config"]["method"] != method or state["step"] != step:
        raise ValueError(f"Unexpected checkpoint method/step: {checkpoint}")
    if state["config"]["data"]["resolution"] != 256:
        raise ValueError("Expected 256-pixel ImageNet reconstruction")
    spec = state["config"]["tokenizer"]
    model = build_tokenizer(spec).eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    identity = {"checkpoint": str(checkpoint), "step": step,
                "model_sha256": state_fingerprint(model), "tokenizer": spec}
    del state
    gc.collect()
    return model, identity


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(output, synset, label, per_class):
    width, top, row_height, pad = 832, 77, 282, 20
    sheet = Image.new("RGB", (width, top + per_class * row_height + 10), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 7), f"{label} ({synset}) | ImageNet validation", fill="black", font=font(20))
    for x, heading in zip((20, 288, 556), ("Original", "B: 10k steps", "C: 4 pool updates")):
        draw.text((x, 42), heading, fill="black", font=font(16))
    for index in range(per_class):
        y = top + index * row_height
        for x, arm in zip((20, 288, 556), ("original", "B", "C")):
            with Image.open(output / synset / arm / f"{index:02d}.png") as image:
                if image.size != (256, 256):
                    raise ValueError("Unexpected reconstruction shape")
                sheet.paste(image.convert("RGB"), (x, y))
        draw.text((pad, y + 258), f"Sample {index + 1}", fill="black", font=font(13))
    sheet.save(output / f"{synset}.png", optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--checkpoint-c", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=217)
    args = parser.parse_args()
    if args.per_class < 2:
        raise ValueError("Select at least two samples per class")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    examples = select_examples(args.val_root, args.per_class, args.seed)
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    inputs = []
    for example in examples:
        with Image.open(example["path"]) as image:
            cropped = center_crop(image.convert("RGB"), 256)
            pixels = np.array(cropped, copy=True)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
        inputs.append(tensor)
        directory = output / example["synset"] / "original"
        directory.mkdir(parents=True, exist_ok=True)
        save_tensor(tensor, directory / f"{example['index']:02d}.png")

    identities = {}
    for arm, checkpoint, method, step in (("B", args.checkpoint_b, "advfd_reconstruction", 10000),
                                           ("C", args.checkpoint_c, "ours_current_both", 4)):
        model, identities[arm] = load_model(checkpoint, method, step)
        with torch.no_grad():
            for start in range(0, len(examples), 2):
                batch = torch.stack(inputs[start:start + 2])
                reconstruction = model(batch)
                if reconstruction.shape != batch.shape or not torch.isfinite(reconstruction).all():
                    raise ValueError(f"Invalid {arm} reconstruction")
                for item, image in zip(examples[start:start + 2], reconstruction):
                    directory = output / item["synset"] / arm
                    directory.mkdir(parents=True, exist_ok=True)
                    save_tensor(image, directory / f"{item['index']:02d}.png")
                print(f"{arm}: {min(start + 2, len(examples))}/{len(examples)}", flush=True)
        del model
        gc.collect()

    for synset, label in CLASSES:
        make_sheet(output, synset, label, args.per_class)
    manifest = {"schema": 1, "seed": args.seed, "per_class": args.per_class,
                "count": len(examples), "classes": [{"synset": s, "label": l} for s, l in CLASSES],
                "samples": [{k: v for k, v in item.items() if k != "path"} for item in examples],
                "models": identities,
                "protocol": "same ImageNet val center crop; CPU eval; uint8 round-half-up PNG"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"complete": {"count": len(examples), "sheets": len(CLASSES),
                                    "output": str(output)}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
