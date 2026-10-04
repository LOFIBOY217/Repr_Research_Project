import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from recon_fd.config import training_signature, CANDIDATE_METHODS
from recon_fd.data import ResumableBatchSampler, sequential_loader, selected_dataset
from recon_fd.evaluation.reference import get_reference
from recon_fd.objectives.static_fd import StaticSpace
from recon_fd.objectives.official_fd import OfficialFDStatistics, OfficialStaticFD, precompute_sigma_ref_sqrt
from recon_fd.objectives.adaptive_fd import AdvFD
from recon_fd.objectives.candidate_fd import CandidateFD
from recon_fd.representations import build_representation
from recon_fd.provenance import write_json, state_fingerprint
from .checkpoint import save_checkpoint, restore_checkpoint
from .adversarial import adversarial_g_step
from .gradients import checked_grad_norm
from .candidate import candidate_g_step


def build_objective(config, dataset, device):
    if config["method"] in CANDIDATE_METHODS:
        static, identities = None, {}
        if config["static"]["enabled"]:
            from copy import deepcopy
            baseline = deepcopy(config)
            baseline["method"] = "fd_only"
            static, identities = build_objective(baseline, dataset, device)
        objective = CandidateFD(config, dataset, static).to(device)
        identities["candidate_real_pool"] = objective.real_reference.pool.identity
        return objective, identities
    static = config["static"]
    reference_data = selected_dataset(dataset, static["reference_samples"], config["data"]["seed"], True)
    spaces, identities = {}, {}
    for spec in static["representations"]:
        extractor = build_representation(spec).to(device)
        reference, identity = get_reference(extractor, reference_data, static["reference_cache"],
                                            config["train"]["batch_size"], device, config["data"]["workers"])
        # FD-Loss and AdvFD publish byte-identical static queue/loss kernels.
        state = OfficialFDStatistics(extractor.dimension, static["queue_size"], static["statistics"],
                                     static["ema_beta"])
        spaces[spec["name"]] = StaticSpace(extractor, reference, state.to(device), spec.get("weight", 1),
                                         root_function=precompute_sigma_ref_sqrt)
        identities[spec["name"]] = identity
    objective = OfficialStaticFD(spaces, static["norm_eps"]).to(device)
    if config["method"] == "advfd_reconstruction":
        objective = AdvFD(objective, config["adaptive"]).to(device)
    return objective, identities


@torch.no_grad()
def initialize_statistics(model, objective, dataset, config, device):
    if isinstance(objective, CandidateFD):
        if objective.static is not None:
            initialize_statistics(model, objective.static, dataset, config, device)
        objective.initialize(model, dataset, config["data"]["seed"], config["data"]["workers"])
        return
    subset = selected_dataset(dataset, config["static"]["initialization_samples"], config["data"]["seed"] + 1, True)
    previous_mode = model.training
    model.eval()
    try:
        for batch in sequential_loader(subset, config["train"]["batch_size"], config["data"]["workers"]):
            reconstruction = model(batch["image"].to(device))
            for name, space in objective.spaces.items():
                features = space.extractor(reconstruction)
                space.statistics.accumulate_initial(features)
        for name, space in objective.spaces.items():
            space.statistics.finalize_initialization()
    finally:
        model.train(previous_mode)


def g_step(model, objective, optimizer, images, grad_clip):
    model.train().requires_grad_(True)
    objective.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction = model(images)
    if reconstruction.shape != images.shape or not torch.isfinite(reconstruction).all():
        raise ValueError("Invalid reconstruction shape or values")
    if reconstruction.min().detach() < 0 or reconstruction.max().detach() > 1:
        raise ValueError("Reconstructor violated public [0,1] image contract")
    result = objective(reconstruction)
    result.loss.backward()
    norms = {}
    for name, parameters in model.parameter_groups().items():
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError(f"No gradient reached {name}")
        norms[name] = float(torch.stack([g.detach().float().norm().square() for g in gradients]).sum().sqrt())
    if any(p.grad is not None or p.requires_grad for s in objective.spaces.values() for p in s.extractor.parameters()):
        raise RuntimeError("Static feature extractor was accidentally trainable")
    norm = checked_grad_norm(model.parameters(), grad_clip)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Optimizer produced non-finite parameters; statistics not committed")
    objective.commit(result)
    return {"loss": float(result.loss.detach()), "raw_fd": result.raw, "normalized_fd": result.normalized,
            "grad_norm": float(norm), "group_grad_norm": norms,
            "clipped_pixel_fraction": float(((reconstruction.detach() <= 0) | (reconstruction.detach() >= 1)).float().mean()),
            "statistics_updates": {k: int(v.statistics.updates) for k, v in objective.spaces.items()}}


def run_training(config, model, objective, dataset, reference_identities, device, resume_state=None):
    run = Path(config["train"]["output"])
    if resume_state is None and run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run}. Choose a new run or --resume.")
    if resume_state is not None and (run / "train.jsonl").exists():
        with (run / "train.jsonl").open() as handle:
            recorded_steps = [json.loads(line)["step"] for line in handle if line.strip()]
        if recorded_steps and max(recorded_steps) > resume_state["step"]:
            raise ValueError("Run has records newer than resume checkpoint; use a fresh output directory")
    run.mkdir(parents=True, exist_ok=True)
    signature = training_signature(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["train"]["lr"],
                                  betas=tuple(config["train"]["betas"]), weight_decay=config["train"]["weight_decay"])
    critic_optimizer = None
    if isinstance(objective, (AdvFD, CandidateFD)):
        adaptive = config["adaptive"]
        critic_optimizer = torch.optim.AdamW(objective.critic_parameters(), lr=adaptive["lr"],
                                            betas=tuple(adaptive["betas"]), weight_decay=adaptive["weight_decay"])
        write_json(run / "advfd_parameters.json", objective.parameter_manifest())
        if isinstance(objective, CandidateFD):
            pool = objective.real_reference.pool
            write_json(run / "real_reference_manifest.json", {"identity": pool.identity, "sample_ids": pool.ids})
    if resume_state is None:
        initialize_statistics(model, objective, dataset, config, device)
        start = 0
        write_json(run / "provenance.json", {"config": config, "data": dataset.identity,
                   "references": reference_identities, "tokenizer_initial_sha256": state_fingerprint(model),
                   "torch_version": torch.__version__, "schema": 1})
        save_checkpoint(run / "checkpoints/step_0000000.pt", model, objective, optimizer, 0, signature, dataset.identity, config, critic_optimizer)
    else:
        start = restore_checkpoint(resume_state, model, objective, optimizer, signature, dataset.identity, critic_optimizer)
        if start >= config["train"]["steps"]:
            raise ValueError("Resume checkpoint has already reached the requested step budget")
        if not (run / "provenance.json").exists():
            write_json(run / "provenance.json", {"config": config, "data": dataset.identity,
                       "references": reference_identities, "resumed_from_step": start,
                       "torch_version": torch.__version__, "schema": 1})
    write_json(run / "config.json", config)
    sampler = ResumableBatchSampler(len(dataset), config["train"]["batch_size"], config["data"]["seed"], start, config["train"]["steps"])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config["data"]["workers"],
                        generator=torch.Generator().manual_seed(config["data"]["seed"]), pin_memory=device.type == "cuda")
    try:
        for step, batch in enumerate(loader, start + 1):
            images = batch["image"].to(device)
            if isinstance(objective, CandidateFD):
                metrics = candidate_g_step(model, objective, optimizer, critic_optimizer, images,
                                            config["train"]["grad_clip"], step - 1)
            elif isinstance(objective, AdvFD):
                metrics = adversarial_g_step(model, objective, optimizer, critic_optimizer, images,
                                             config["train"]["grad_clip"], step - 1)
            else:
                metrics = g_step(model, objective, optimizer, images, config["train"]["grad_clip"])
            metrics.update(step=step, samples_seen=step * config["train"]["batch_size"],
                           fd_batch_size=len(batch["id"]), sample_ids=batch["id"])
            with (run / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, allow_nan=False) + "\n")
            if step % config["train"]["log_every"] == 0:
                print(json.dumps(metrics, allow_nan=False), flush=True)
            if step % config["train"]["save_every"] == 0 or step == config["train"]["steps"]:
                save_checkpoint(run / f"checkpoints/step_{step:07d}.pt", model, objective, optimizer,
                                step, signature, dataset.identity, config, critic_optimizer)
    except Exception as error:
        write_json(run / "failure.json", {"error": repr(error), "last_attempted_step": locals().get("step", start),
                                          "resume_from": "last completed checkpoint, not partially updated state"})
        raise
    return run
