"""Real-reference estimators for C; image identity and feature version differ."""
from dataclasses import dataclass
import torch
from torch import nn
from recon_fd.data import sequential_loader
from .adaptive_fd import AdvStatsEMA
from .statistics import Moments, RunningMoments


@dataclass
class RealReferenceView:
    moments: Moments
    pending: torch.Tensor | None
    estimator_version: int
    loss_psi_version: int
    feature_psi_version: int


class RealReference(nn.Module):
    def __init__(self, dimension, config, beta, pool, workers=0):
        super().__init__()
        self.mode = config["mode"]
        self.batch_size = config["batch_size"]
        self.pool, self.workers = pool, workers
        self.ema = AdvStatsEMA(dimension, beta) if self.mode == "ema" else None
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer("covariance", torch.zeros(dimension, dimension, dtype=torch.float64))
        self.register_buffer("feature_version", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("refreshes", torch.tensor(0, dtype=torch.long))
        self.register_buffer("feature_images", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def refresh(self, extractor, psi_version):
        accumulator = RunningMoments()
        for batch in sequential_loader(self.pool, self.batch_size, self.workers):
            accumulator.update(extractor(batch["image"].to(self.mean.device)))
        # Sample covariance, like the fixed reference used to initialize B.
        moments = accumulator.moments(ddof=1)
        self.mean.copy_(moments.mean)
        self.covariance.copy_(moments.cov)
        self.feature_version.fill_(psi_version)
        self.refreshes.add_(1)
        self.feature_images.add_(moments.count)
        self.initialized.fill_(True)
        return moments

    @torch.no_grad()
    def initialize(self, extractor, psi_version):
        if self.initialized:
            raise RuntimeError("Real reference already initialized")
        moments = self.refresh(extractor, psi_version)
        if self.ema is not None:
            self.ema.initialize_mean_cov(moments.mean, moments.cov)

    def preview(self, extractor, images, psi_version):
        if not self.initialized:
            raise RuntimeError("Real reference not initialized")
        if self.mode == "reencode_pool":
            if int(self.feature_version) != psi_version:
                self.refresh(extractor, psi_version)
            # Own snapshots: a later cache refresh cannot mutate a loss graph.
            moments = Moments(self.mean.detach().clone(), self.covariance.detach().clone(), len(self.pool))
            return RealReferenceView(moments, None, 0, psi_version, int(self.feature_version))
        if self.mode == "frozen_initial":
            moments = Moments(self.mean.detach().clone(), self.covariance.detach().clone(), len(self.pool))
            return RealReferenceView(moments, None, 0, psi_version, int(self.feature_version))
        with torch.no_grad():
            features = extractor(images.detach())
        if self.mode == "ema":
            # Deliberately a historical mixture, NOT exact current-psi moments.
            return RealReferenceView(self.ema.preview(features), features.detach(),
                                     int(self.ema.updates), psi_version, psi_version)
        raise ValueError(f"Unsupported real-reference mode: {self.mode}")

    def validate_pending(self, view, psi_version):
        if view.loss_psi_version != psi_version:
            raise RuntimeError("Stale psi version in real reference")
        if self.mode == "reencode_pool" and view.feature_psi_version != psi_version:
            raise RuntimeError("Stale re-encoded real statistics")
        if self.ema is not None and view.estimator_version != int(self.ema.updates):
            raise RuntimeError("Stale or duplicate real EMA commit")

    @torch.no_grad()
    def commit(self, view, psi_version):
        self.validate_pending(view, psi_version)
        if self.ema is not None:
            self.ema.commit(view.pending, view.estimator_version)
