import copy
import json
import os
from pathlib import Path
import yaml
from .provenance import fingerprint

CANDIDATE_METHODS = {"ours", "ours_add_static", "ours_lora", "ours_real_ema", "ours_fixed_reference", "ours_current_both"}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path, overrides=(), _seen=()):
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError("Configuration inheritance cycle")
    with path.open() as handle:
        # PyYAML treats JSON's valid exponent-only floats (e.g. 1e-06) as strings.
        config = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    parent = config.pop("extends", None)
    if parent:
        config = merge(load_config(path.parent / parent, _seen=(*_seen, path)), config)
    for override in overrides:
        key, value = override.split("=", 1)
        current = config
        parts = key.split(".")
        for part in parts[:-1]:
            current = current[part]
        if parts[-1] not in current:
            raise ValueError(f"Unknown override: {key}")
        try:
            current[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            current[parts[-1]] = yaml.safe_load(value)
    return config


def expand_path(value):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if "${" in value or "$" in value:
        raise ValueError(f"Unset environment variable in path: {value}")
    return str(Path(value).resolve())


def validate(config):
    expected = {"schema_version", "method", "runtime", "data", "tokenizer", "static", "adaptive", "train", "evaluation"}
    if set(config) != expected or config["schema_version"] != 1:
        raise ValueError(f"Unexpected config sections: {set(config) ^ expected}")
    allowed = {
        "runtime": {"device", "seed", "threads"},
        "data": {"kind", "resolution", "train_path", "val_path", "workers", "seed", "synthetic_count"},
        "tokenizer": {"kind", "checkpoint", "local_files_only", "gradient_checkpointing", "trainable_scope"},
        "static": {"enabled", "statistics", "ema_beta", "queue_size", "reference_samples", "initialization_samples", "norm_eps", "reference_cache", "representations"},
        "adaptive": {"enabled", "trainable_scope", "real_stats"},
        "train": {"output", "steps", "batch_size", "lr", "weight_decay", "betas", "grad_clip", "grad_accumulation", "save_every", "log_every"},
        "evaluation": {"num_samples", "batch_size", "engineering_only", "paired_metrics", "representations"},
    }
    if config["method"] == "advfd_reconstruction" or config["method"] in CANDIDATE_METHODS:
        allowed["adaptive"] |= {"representation", "weight", "ema_beta", "whiten_eps", "lr", "betas",
                                "weight_decay", "grad_clip", "start_step", "warmup_steps", "update_freq",
                                "steps_per_update", "lora", "gradient_checkpointing"}
    if config["method"] in CANDIDATE_METHODS:
        allowed["adaptive"] |= {"norm_eps", "initialization_samples", "initialization_batch_size"}
    if config["method"] == "ours_current_both":
        allowed["adaptive"] |= {"fake_stats"}
    for section, keys in allowed.items():
        if set(config[section]) != keys:
            raise ValueError(f"Unknown or missing keys in {section}: {set(config[section]) ^ keys}")
    if config["method"] not in {"fd_only", "advfd_reconstruction"} | CANDIDATE_METHODS:
        raise NotImplementedError("Unsupported reconstruction method")
    if config["adaptive"]["enabled"] != (config["method"] != "fd_only"):
        raise ValueError("Method and adaptive.enabled disagree")
    if config["tokenizer"]["trainable_scope"] != "encoder_decoder":
        raise ValueError("All methods require joint encoder+decoder training")
    if config["method"] not in CANDIDATE_METHODS and not config["static"]["enabled"]:
        raise ValueError("Both baselines require static FD")
    if config["static"]["statistics"] not in {"ema", "queue", "queue_online"}:
        raise ValueError("static.statistics must be ema, queue or queue_online")
    if not 0 <= config["static"]["ema_beta"] < 1 or config["static"]["norm_eps"] <= 0:
        raise ValueError("Invalid EMA beta or loss-normalization epsilon")
    for section, keys in (("train", ("steps", "batch_size", "save_every", "log_every")),
                          ("static", ("reference_samples", "initialization_samples", "queue_size")),
                          ("evaluation", ("num_samples", "batch_size"))):
        for key in keys:
            if not isinstance(config[section][key], int) or config[section][key] < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    if config["train"]["batch_size"] < 2:
        raise ValueError("FD training batch must contain >=2 images")
    if config["train"]["grad_accumulation"] != 1:
        raise NotImplementedError("Gradient accumulation is not a pooled FD batch; first version requires 1")
    for key in ("lr", "grad_clip"):
        value = config["train"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"train.{key} must be a number, got {type(value).__name__}")
    if not 0 < config["train"]["lr"] < float("inf") or not 0 <= config["train"]["grad_clip"] < float("inf"):
        raise ValueError("Positive learning rate and nonnegative finite gradient clipping are required")
    if config["data"]["resolution"] < 8:
        raise ValueError("Image resolution must be >=8")
    for field in ("representations",):
        for section in ("static", "evaluation"):
            reps = config[section][field]
            names = [r["name"] for r in reps]
            if ((not reps and (section == "evaluation" or config["static"]["enabled"]))
                    or len(names) != len(set(names))):
                raise ValueError("Representation names must be nonempty and unique")
            for spec in reps:
                if set(spec) - {"name", "kind", "weights", "pool", "weight", "seed", "model_name", "target_size"}:
                    raise ValueError("Unknown representation config keys")
                if not spec["name"].replace("_", "").isalnum():
                    raise ValueError("Representation names must be alphanumeric/underscore")
                if spec.get("pool", "cls") not in {"cls", "avg"}:
                    raise ValueError("Unsupported feature pooling")
                if spec.get("weight", 1.0) <= 0:
                    raise ValueError("FD weights must be positive")
                if spec["kind"] == "inception" and (spec.get("pool", "cls") != "cls" or spec.get("target_size", 299) != 299):
                    raise ValueError("Official Inception uses pool_2048 with internal TF resize to 299")
    if config["static"]["enabled"]:
        if config["static"]["initialization_samples"] != config["static"]["queue_size"]:
            raise ValueError("Official static FD: initialization_samples must equal queue_size")
        if config["static"]["norm_eps"] != 0.01:
            raise ValueError("Static FD keeps the official recipe's normalization epsilon 0.01")
        if config["static"]["statistics"] == "ema" and config["static"]["ema_beta"] <= 0:
            raise ValueError("Official EMA requires positive beta; use queue for beta=0")
    if config["static"]["statistics"] in {"queue", "queue_online"}:
        if config["static"]["initialization_samples"] != config["static"]["queue_size"]:
            raise ValueError("Queue must be initialized with exactly queue_size observations")
        if config["train"]["batch_size"] > config["static"]["queue_size"]:
            raise ValueError("Batch exceeds queue capacity")
    if not config["evaluation"]["engineering_only"]:
        if config["evaluation"]["num_samples"] < 50000:
            raise ValueError("Scientific evaluation requires at least 50,000 images")
        if config["data"]["kind"] != "imagenet" or config["tokenizer"]["kind"] == "tiny":
            raise ValueError("Synthetic/tiny models are engineering-only")
        if any(s["kind"] == "tiny" for s in config["static"]["representations"] + config["evaluation"]["representations"]):
            raise ValueError("Tiny features are engineering-only")
    if config["runtime"]["device"] not in {"cpu", "cuda"}:
        raise ValueError("Use CPU or one CUDA GPU; float64 FD on MPS is not supported")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This version is explicitly single-process, single-GPU")
    if config["method"] == "advfd_reconstruction":
        validate_advfd(config)
    elif config["method"] in CANDIDATE_METHODS:
        validate_candidate(config)


def validate_candidate(config):
    method, adaptive = config["method"], config["adaptive"]
    if config["static"]["enabled"] != (method == "ours_add_static"):
        raise ValueError("Only the explicitly labelled ours_add_static ablation enables static FD")
    if not config["static"]["enabled"] and config["static"]["representations"]:
        raise ValueError("C must have no hidden static representations")
    scope = "lora" if method == "ours_lora" else "full"
    if adaptive["trainable_scope"] != scope:
        raise ValueError(f"{method} requires {scope} scope")
    mode = {"ours_real_ema": "ema", "ours_fixed_reference": "frozen_initial"}.get(method, "reencode_pool")
    real = adaptive["real_stats"]
    if set(real) != {"mode", "samples", "batch_size", "seed"} or real["mode"] != mode:
        raise ValueError(f"{method} requires real_stats mode {mode} and an explicit image-pool configuration")
    for key in ("samples", "batch_size"):
        if not isinstance(real[key], int) or real[key] < (2 if key == "samples" else 1):
            raise ValueError(f"Invalid real_stats.{key}")
    if not isinstance(real["seed"], int) or real["seed"] < 0:
        raise ValueError("Invalid real pool seed")
    for key in ("initialization_samples", "initialization_batch_size", "update_freq", "steps_per_update"):
        if not isinstance(adaptive[key], int) or adaptive[key] < (2 if key == "initialization_samples" else 1):
            raise ValueError(f"Invalid adaptive.{key}")
    if method == "ours_current_both":
        if adaptive["fake_stats"] != {"mode": "reencode_pool", "gradient": "full_pool_replay"}:
            raise ValueError("Current-both C requires fresh full-pool moments and full-pool replay gradients")
        if adaptive["initialization_samples"] != real["samples"]:
            raise ValueError("Current-both C uses the same paired pool for real and reconstructed images")
        if adaptive["initialization_batch_size"] != config["train"]["batch_size"]:
            raise ValueError("Current-both C uses train.batch_size for all reconstruction microbatches")
    if adaptive["start_step"] != 0 or adaptive["warmup_steps"] != 0:
        raise ValueError("C starts after statistics initialization with nonzero loss; no static-only warmup")
    for key in ("lr", "weight", "whiten_eps", "grad_clip"):
        if not isinstance(adaptive[key], (int, float)) or not 0 < adaptive[key] < float("inf"):
            raise ValueError(f"Invalid adaptive.{key}")
    if (adaptive["norm_eps"] != 0.01 or not 0 < adaptive["ema_beta"] < 1
            or not 0 <= adaptive["weight_decay"] < float("inf")):
        raise ValueError("Invalid candidate normalization/EMA/weight decay")
    if len(adaptive["betas"]) != 2 or any(not 0 <= b < 1 for b in adaptive["betas"]):
        raise ValueError("Invalid candidate optimizer betas")
    spec = adaptive["representation"]
    if not isinstance(spec, dict) or not {"name", "kind"} <= set(spec):
        raise ValueError("C needs an independent representation specification, not a static-branch name")
    if set(spec) - {"name", "kind", "weights", "pool", "weight", "seed", "model_name", "target_size"}:
        raise ValueError("Unknown candidate representation keys")
    if not isinstance(spec["name"], str) or not spec["name"].replace("_", "").isalnum():
        raise ValueError("Candidate representation name must be alphanumeric/underscore")
    if spec["kind"] == "timm" and not spec.get("model_name"):
        raise ValueError("Candidate timm representation requires model_name")
    if spec["kind"] not in {"inception", "timm", "tiny"} or spec.get("pool", "cls") not in {"cls", "avg"}:
        raise ValueError("Unsupported candidate representation")
    if spec["kind"] == "inception" and (spec.get("pool", "cls") != "cls" or spec.get("target_size", 299) != 299):
        raise ValueError("Candidate Inception retains the official pool_2048/299 convention")
    if adaptive["gradient_checkpointing"] and spec["kind"] != "timm":
        raise ValueError("Candidate representation checkpointing currently supports timm only")
    lora = adaptive["lora"]
    if (set(lora) != {"rank", "alpha", "targets", "dropout"} or lora["rank"] != 16
            or lora["alpha"] != 16 or lora["targets"] != ["attn.qkv"] or lora["dropout"] != 0):
        raise ValueError("LoRA ablation retains B's rank-16 QKV configuration")
    if scope == "lora" and spec["kind"] != "timm":
        raise ValueError("LoRA ablation requires the same timm backbone as the full-tuning comparison")
    if not config["evaluation"]["engineering_only"]:
        if real["samples"] < 50000 or adaptive["initialization_samples"] < 50000:
            raise ValueError("Formal C runs require 50,000 reference and initialization images")
        if spec["kind"] == "tiny":
            raise ValueError("Tiny candidate representation is engineering-only")
        if spec["kind"] == "timm" and spec.get("model_name") not in {
                "vit_large_patch16_224.mae", "vit_so400m_patch16_siglip_256.v2_webli"}:
            raise ValueError("First C release supports the matched MAE/SigLIP scientific backbones")


def validate_advfd(config):
    """Guard B from silently turning into any of the candidate-method ablations."""
    adaptive = config["adaptive"]
    if adaptive["trainable_scope"] != "paper":
        raise ValueError("B uses paper scope: full Inception, rank-16 LoRA for SigLIP/MAE")
    real = adaptive["real_stats"]
    if set(real) != {"mode", "update_freq"} or real["mode"] != "ema":
        raise ValueError("B retains AdvFD real-reference EMA; other estimators are not B")
    if config["static"]["statistics"] != "ema":
        raise ValueError("B paper recipes require static feature-statistics EMA")
    if config["static"]["norm_eps"] != 0.01:
        raise ValueError("B keeps the upstream FD normalization epsilon 0.01")
    representations = {spec["name"]: spec for spec in config["static"]["representations"]}
    if adaptive["representation"] not in representations:
        raise ValueError("B needs a matching static representation to initialize adv fake moments")
    spec = representations[adaptive["representation"]]
    supported = {"vit_large_patch16_224.mae", "vit_so400m_patch16_siglip_256.v2_webli"}
    if spec["kind"] == "timm" and spec.get("model_name") not in supported:
        raise ValueError("B paper backbone must be Inception, SigLIP or MAE")
    if spec["kind"] not in {"inception", "timm", "tiny"}:
        raise ValueError("Unsupported B representation")
    if spec["kind"] == "tiny" and not config["evaluation"]["engineering_only"]:
        raise ValueError("Tiny adversarial branch is engineering-only")
    lora = adaptive["lora"]
    if (set(lora) != {"rank", "alpha", "dropout", "targets"} or lora["rank"] != 16
            or lora["targets"] != ["attn.qkv"] or lora["alpha"] <= 0 or lora["dropout"] != 0):
        raise ValueError("B uses paper rank-16 LoRA; QKV targets/dropout follow the released recipes")
    for key in ("lr", "weight", "whiten_eps", "grad_clip"):
        if not isinstance(adaptive[key], (int, float)) or not 0 < adaptive[key] < float("inf"):
            raise ValueError(f"Invalid adaptive.{key}")
    for key in ("start_step", "warmup_steps", "update_freq", "steps_per_update"):
        minimum = 0 if key in {"warmup_steps", "start_step"} else 1
        if not isinstance(adaptive[key], int) or adaptive[key] < minimum:
            raise ValueError(f"Invalid adaptive.{key}")
    if not isinstance(real["update_freq"], int) or real["update_freq"] < 1:
        raise ValueError("Invalid real-statistics update frequency")
    if not 0 <= adaptive["ema_beta"] < 1 or not 0 <= adaptive["weight_decay"] < float("inf"):
        raise ValueError("Invalid adversarial EMA or optimizer weight decay")
    if len(adaptive["betas"]) != 2 or any(not 0 <= b < 1 for b in adaptive["betas"]):
        raise ValueError("Invalid adversarial optimizer betas")


def training_signature(config):
    value = copy.deepcopy(config)
    # A resumed run can extend its budget or change reporting, not its science.
    for key in ("steps", "save_every", "log_every", "output"):
        value["train"].pop(key)
    value.pop("evaluation")
    value["runtime"].pop("threads", None)
    return fingerprint(value)
