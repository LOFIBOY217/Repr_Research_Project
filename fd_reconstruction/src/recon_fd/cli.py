import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from .config import load_config, validate, expand_path, training_signature, CANDIDATE_METHODS
from .data import build_dataset, selected_dataset
from .tokenizers import build_tokenizer
from .representations import build_representation
from .engine.trainer import build_objective, run_training
from .engine.checkpoint import read_checkpoint
from .evaluation.reference import get_reference
from .evaluation.runner import export_reconstructions, evaluate_export
from .runtime import configure_determinism


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser


def setup(args):
    config = load_config(args.config, args.set)
    validate(config)
    device = torch.device(config["runtime"]["device"])
    configure_determinism(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use configs/smoke.yaml for offline CPU tests")
    torch.set_num_threads(config["runtime"]["threads"])
    random.seed(config["runtime"]["seed"])
    np.random.seed(config["runtime"]["seed"])
    torch.manual_seed(config["runtime"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["runtime"]["seed"])
    # Resolve paths into provenance. No implicit dependency on sibling repos.
    config["train"]["output"] = expand_path(config["train"]["output"])
    config["static"]["reference_cache"] = expand_path(config["static"]["reference_cache"])
    if config["data"]["kind"] == "imagenet":
        for key in ("train_path", "val_path"):
            config["data"][key] = expand_path(config["data"][key])
    checkpoint = config["tokenizer"]["checkpoint"]
    if checkpoint and ("$" in checkpoint or Path(checkpoint).exists()):
        config["tokenizer"]["checkpoint"] = expand_path(checkpoint)
    representations = config["static"]["representations"] + config["evaluation"]["representations"]
    if config["method"] in CANDIDATE_METHODS:
        representations = representations + [config["adaptive"]["representation"]]
    for spec in representations:
        if spec.get("weights"):
            spec["weights"] = expand_path(spec["weights"])
    return config, device


def train_main(argv=None):
    parser = common_parser("FD-only, AdvFD or candidate image reconstruction post-training")
    parser.add_argument("--resume", help="Trusted local complete checkpoint; never load untrusted pickle files")
    args = parser.parse_args(argv)
    config, device = setup(args)
    state = read_checkpoint(args.resume) if args.resume else None
    if state and state["signature"] != training_signature(config):
        raise ValueError("Resume configuration differs from checkpoint")
    dataset = build_dataset(config, "train")
    model = build_tokenizer(config["tokenizer"]).to(device)
    objective, references = build_objective(config, dataset, device)
    run = run_training(config, model, objective, dataset, references, device, state)
    print(f"Completed {config['method']} run: {run}")


def evaluate_main(argv=None):
    parser = common_parser("Export and evaluate image reconstructions using fixed independent representations")
    parser.add_argument("--checkpoint", help="Trusted local checkpoint; omit to evaluate the unmodified tokenizer")
    parser.add_argument("--output", required=True, help="Fresh evaluation directory or matching completed export")
    args = parser.parse_args(argv)
    config, device = setup(args)
    dataset = selected_dataset(build_dataset(config, "val"), config["evaluation"]["num_samples"], config["data"]["seed"])
    model = build_tokenizer(config["tokenizer"]).to(device)
    state = read_checkpoint(args.checkpoint) if args.checkpoint else None
    if state:
        if state["config"]["tokenizer"] != config["tokenizer"]:
            raise ValueError("Evaluation tokenizer does not match checkpoint configuration")
        model.load_state_dict(state["model"], strict=True)
    output = Path(expand_path(args.output))
    export = export_reconstructions(model, dataset, output / "reconstructions", config, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = evaluate_export(dataset, output / "reconstructions", output, config, device)
    results["checkpoint_step"] = state["step"] if state else 0
    results["checkpoint_model_sha256"] = export["identity"]["model_sha256"]
    from .provenance import write_json
    write_json(output / "metrics.json", results)
    print(json.dumps(results["fd"], indent=2))


def reference_main(argv=None):
    parser = common_parser("Prepare fingerprinted fixed real reference statistics")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args(argv)
    config, device = setup(args)
    if args.split == "train" and not config["static"]["enabled"]:
        raise ValueError("C real statistics are versioned and prepared by training, not a frozen reference file")
    count = config["static"]["reference_samples"] if args.split == "train" else config["evaluation"]["num_samples"]
    dataset = selected_dataset(build_dataset(config, args.split), count, config["data"]["seed"], args.split == "train")
    representations = config["static" if args.split == "train" else "evaluation"]["representations"]
    for spec in representations:
        extractor = build_representation(spec).to(device)
        _, identity = get_reference(extractor, dataset, config["static"]["reference_cache"],
                                    config["evaluation"]["batch_size"], device, config["data"]["workers"])
        print(f"Prepared {args.split} reference: {spec['name']} ({count} images)")
        del extractor
