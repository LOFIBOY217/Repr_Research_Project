"""A baseline: call the unchanged FD-Loss core; adapt reconstruction I/O only.

Initialization and forward dispatch follow upstream judges.fill_all_queues and
main_fd.get_fd_train_step. Keep upstream dtypes, eigvalsh/sqrt derivatives and
FP32 loss return; do not route A through our stabilized frechet.py implementation.
The wrapper adds lifecycle validation and a single post-optimizer commit. On
successful steps this enqueues the same detached pre-update features as upstream.
"""
import torch
from torch import nn
from recon_fd.vendor.fd_loss.queue import FeatureQueue
from recon_fd.vendor.fd_loss.losses import (
    compute_frechet_distance_loss, diff_all_gather, precompute_sigma_ref_sqrt,
)
from .statistics import check_features
from .static_fd import LossResult, StaticFD


class OfficialFDStatistics(nn.Module):
    def __init__(self, dimension, capacity, mode, beta):
        super().__init__()
        if mode not in {"ema", "queue", "queue_online"} or capacity < 2:
            raise ValueError("Invalid official FD statistics configuration")
        if mode == "ema" and not 0 < beta < 1:
            raise ValueError("Official EMA requires positive beta; beta=0 selects a queue upstream")
        self.queue = FeatureQueue(size=capacity, feat_dim=dimension,
                                  online_accum=mode == "queue_online",
                                  ema_beta=beta if mode == "ema" else 0.0)
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def accumulate_initial(self, features):
        check_features(features)
        start = int(self.filled)
        if self.initialized or start + len(features) > self.queue.size:
            raise ValueError("Official initialization must fill exactly queue_size features once")
        if self.queue.ema_stats:
            self.queue.accumulate_batch(features)
        else:
            self.queue.feats[start:start + len(features)] = features.detach().float()
        self.filled.add_(len(features))

    @torch.no_grad()
    def finalize_initialization(self):
        if self.initialized or int(self.filled) != self.queue.size:
            raise ValueError("Incomplete or duplicate official initialization")
        if self.queue.ema_stats:
            self.queue._finalize_streaming_init()
        else:
            self.queue.ptr.zero_()
            if self.queue.online_accum:
                self.queue._init_accumulators()
        self.initialized.fill_(True)

    def fd(self, features, reference_mean, reference_cov, reference_sqrt):
        check_features(features)
        if not self.initialized:
            raise RuntimeError("Initialize the official FD statistics before training")
        if not self.queue.ema_stats and len(features) > self.queue.size:
            raise ValueError("Batch exceeds official queue capacity")
        kwargs = {"sigma_ref_sqrt": reference_sqrt}
        if self.queue.online_accum or self.queue.ema_stats:
            mu, sigma = self.queue.build_feats_stats(features)
            kwargs.update(mu=mu, sigma=sigma)
        else:
            kwargs["all_feats"] = self.queue.build_feats_snapshot(features)
        return compute_frechet_distance_loss(reference_mean, reference_cov, **kwargs)

    @torch.no_grad()
    def commit(self, features, expected_version):
        if not self.initialized or int(self.updates) != expected_version:
            raise RuntimeError("Stale or duplicate official statistics commit")
        check_features(features)
        self.queue.enqueue(features.detach())
        self.updates.add_(1)


class OfficialStaticFD(StaticFD):
    """Upstream FD dispatch and normalization with the common engine interface."""
    def __init__(self, spaces, norm_eps=0.01):
        super().__init__(spaces, norm_eps)
        if norm_eps != 0.01:
            raise ValueError("A retains the official recipe's normalization epsilon 0.01")
        for space in self.spaces.values():
            if not isinstance(space.statistics, OfficialFDStatistics):
                raise TypeError("A must use the unchanged official statistics backend")

    def forward(self, reconstructions):
        total = torch.tensor(0.0, device=reconstructions.device)
        raw, normalized, pending = {}, {}, {}
        for name, space in self.spaces.items():
            features = diff_all_gather(space.extractor(reconstructions))
            fd = space.statistics.fd(features, space.reference_mean,
                                     space.reference_cov, space.reference_sqrt)
            if not torch.isfinite(fd):
                raise FloatingPointError("Non-finite official FD; stop instead of substituting a loss")
            loss = fd / (fd.detach() + self.norm_eps)
            total = total + space.weight * loss
            raw[name], normalized[name] = float(fd.detach()), float(loss.detach())
            pending[name] = (features.detach(), int(space.statistics.updates))
        return LossResult(total, raw, normalized, pending)
