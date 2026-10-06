import json
from pathlib import Path
import numpy as np
import torch
from recon_fd.objectives.statistics import RunningMoments, Moments
from recon_fd.provenance import fingerprint
from recon_fd.data import sequential_loader
from .progress import log_progress, should_log


def reference_identity(extractor, dataset, pixel_protocol="float01"):
    return {"schema": 1, "representation": extractor.identity, "data": dataset.identity,
            "pixel_protocol": pixel_protocol, "covariance_ddof": 1}


def reference_path(cache_dir, identity):
    return Path(cache_dir) / (fingerprint(identity) + ".npz")


def save_reference(path, moments, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".tmp").open("wb") as handle:
        np.savez(handle, mean=moments.mean.detach().cpu().numpy(), cov=moments.cov.detach().cpu().numpy(),
                 count=np.asarray(moments.count), metadata=np.asarray(json.dumps(identity, sort_keys=True)))
    path.with_suffix(".tmp").replace(path)


def load_reference(path, identity, device):
    with np.load(path, allow_pickle=False) as values:
        if json.loads(str(values["metadata"].item())) != identity:
            raise ValueError("Reference provenance mismatch; regenerate instead of guessing compatibility")
        mean = torch.as_tensor(values["mean"].copy(), device=device, dtype=torch.float64)
        cov = torch.as_tensor(values["cov"].copy(), device=device, dtype=torch.float64)
        count = int(values["count"])
    if count != identity["data"]["count"] or count < 2 or not torch.isfinite(mean).all() or not torch.isfinite(cov).all():
        raise ValueError("Corrupt or incomplete reference statistics")
    return Moments(mean, cov, count)


@torch.no_grad()
def get_reference(extractor, dataset, cache_dir, batch_size, device, workers=0):
    identity = reference_identity(extractor, dataset)
    path = reference_path(cache_dir, identity)
    if path.exists():
        log_progress(f"reference {extractor.identity['spec']['name']} cached", len(dataset), len(dataset))
        return load_reference(path, identity, device), identity
    accumulator = RunningMoments()
    seen = 0
    for batch in sequential_loader(dataset, batch_size, workers):
        accumulator.update(extractor(batch["image"].to(device)))
        previous, seen = seen, seen + len(batch["id"])
        if should_log(seen, previous, len(dataset)):
            log_progress(f"reference {extractor.identity['spec']['name']}", seen, len(dataset))
    moments = accumulator.moments()
    save_reference(path, moments, identity)
    return moments, identity
