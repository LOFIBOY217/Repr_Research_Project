from dataclasses import dataclass
import torch
from torch import nn
from .frechet import frechet_distance, covariance_sqrt


@dataclass
class LossResult:
    loss: torch.Tensor
    raw: dict
    normalized: dict
    pending: dict


class StaticSpace(nn.Module):
    def __init__(self, extractor, reference, statistics, weight=1.0, root_function=covariance_sqrt):
        super().__init__()
        self.extractor = extractor.eval().requires_grad_(False)
        self.statistics = statistics
        self.weight = float(weight)
        self.register_buffer("reference_mean", reference.mean.detach().double().clone())
        self.register_buffer("reference_cov", reference.cov.detach().double().clone())
        self.register_buffer("reference_sqrt", root_function(self.reference_cov))

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()
        return self


class StaticFD(nn.Module):
    """No forward-side mutations. Gradients flow through frozen extractors."""
    def __init__(self, spaces, norm_eps=0.01):
        super().__init__()
        if not spaces or norm_eps <= 0:
            raise ValueError("Need at least one FD space and positive norm_eps")
        self.spaces = nn.ModuleDict(spaces)
        self.norm_eps = float(norm_eps)

    def forward(self, reconstructions):
        total, raw, normalized, pending = None, {}, {}, {}
        for name, space in self.spaces.items():
            features = space.extractor(reconstructions)
            moments = space.statistics.preview(features)
            fd = frechet_distance(space.reference_mean, space.reference_cov,
                                  moments.mean, moments.cov, space.reference_sqrt)
            loss = fd / (fd.detach() + self.norm_eps)
            term = space.weight * loss
            total = term if total is None else total + term
            raw[name], normalized[name] = float(fd.detach()), float(loss.detach())
            pending[name] = (features.detach(), int(space.statistics.updates))
        return LossResult(total, raw, normalized, pending)

    @torch.no_grad()
    def commit(self, result):
        # Validate every branch before mutating any branch.
        for name, (_, version) in result.pending.items():
            if int(self.spaces[name].statistics.updates) != version:
                raise RuntimeError("Duplicate or stale loss result")
        for name, (features, version) in result.pending.items():
            self.spaces[name].statistics.commit(features, version)
