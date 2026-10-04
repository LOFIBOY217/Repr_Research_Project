"""Finite-pool current-model moments: no historical features on either side."""
from dataclasses import dataclass
import torch
from torch import nn
from recon_fd.data import sequential_loader
from .candidate_fd import CandidateFD
from .real_reference import RealReferenceView
from .statistics import Moments, RunningMoments


def checked_reconstruction(model, images):
    reconstruction = model(images)
    if (reconstruction.shape != images.shape or not torch.isfinite(reconstruction).all()
            or reconstruction.min().detach() < 0 or reconstruction.max().detach() > 1):
        raise ValueError("Invalid reconstruction")
    return reconstruction


@dataclass
class CurrentPair:
    real: RealReferenceView
    fake: Moments
    psi_version: int
    generator_version: int


class ReconstructionReference(nn.Module):
    """Cache valid only for the exact (psi, E+D) parameter versions."""
    def __init__(self, dimension):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer("covariance", torch.zeros(dimension, dimension, dtype=torch.float64))
        for name, initial in (("psi_version", -1), ("generator_version", -1),
                              ("refreshes", 0), ("feature_images", 0)):
            self.register_buffer(name, torch.tensor(initial, dtype=torch.long))

    def current(self, psi_version, generator_version):
        return int(self.psi_version) == psi_version and int(self.generator_version) == generator_version

    @torch.no_grad()
    def refresh(self, model, extractor, pool, batch_size, workers, psi_version, generator_version):
        if model.training or extractor.training:
            raise RuntimeError("Current-pool recomputation requires deterministic eval-mode models")
        accumulator = RunningMoments()
        for batch in sequential_loader(pool, batch_size, workers):
            images = batch["image"].to(self.mean.device)
            accumulator.update(extractor(checked_reconstruction(model, images)))
        moments = accumulator.moments(ddof=1)
        self.mean.copy_(moments.mean)
        self.covariance.copy_(moments.cov)
        self.psi_version.fill_(psi_version)
        self.generator_version.fill_(generator_version)
        self.refreshes.add_(1)
        self.feature_images.add_(moments.count)

    def snapshot(self, count):
        return Moments(self.mean.detach().clone(), self.covariance.detach().clone(), count)


class CurrentBothFD(CandidateFD):
    """Explicit C extension; the earlier C+fake-EMA route remains unchanged."""
    def build_fake_statistics(self, dimension):
        return None  # Do not even construct an EMA or feature queue.

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.fake_reference = ReconstructionReference(self.extractor.dimension)
        self.microbatch_size = config["train"]["batch_size"]
        self.register_buffer("generator_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("gradient_feature_images", torch.tensor(0, dtype=torch.long))

    def parameter_manifest(self):
        result = super().parameter_manifest()
        result.update(fake_statistics="current_model_reencoded_pool", fake_covariance_ddof=1,
                      gradient_estimator="full_pool_two_pass_chain_rule", paired_pool=True,
                      tokenizer_mode="eval_with_grad", microbatch_size=self.microbatch_size)
        return result

    @torch.no_grad()
    def initialize(self, model, dataset, seed, workers):
        if self.initialized:
            raise RuntimeError("Candidate statistics already initialized")
        previous = model.training
        model.eval()
        try:
            self.real_reference.initialize(self.extractor, int(self.critic_updates))
            self._refresh_fake(model)
        finally:
            model.train(previous)
        self.initialized.fill_(True)

    def _refresh_fake(self, model):
        self.fake_reference.refresh(model, self.extractor, self.real_reference.pool,
                                    self.microbatch_size, self.real_reference.workers,
                                    int(self.critic_updates), int(self.generator_updates))

    def current_pair(self, model):
        if not self.initialized:
            raise RuntimeError("Candidate statistics not initialized")
        psi, generator = int(self.critic_updates), int(self.generator_updates)
        real = self.real_reference.preview(self.extractor, None, psi)
        if not self.fake_reference.current(psi, generator):
            self._refresh_fake(model)
        return CurrentPair(real, self.fake_reference.snapshot(len(self.real_reference.pool)), psi, generator)

    def validate_pair(self, pair):
        if pair.psi_version != int(self.critic_updates):
            raise RuntimeError("Stale psi version in current-pool loss")
        if pair.generator_version != int(self.generator_updates):
            raise RuntimeError("Stale E+D version in current-pool loss")
        self.real_reference.validate_pending(pair.real, int(self.critic_updates))
        if not self.fake_reference.current(pair.psi_version, pair.generator_version):
            raise RuntimeError("Stale reconstruction reference cache")

    def dynamic(self, *args, **kwargs):
        raise RuntimeError("CurrentBothFD requires the full-pool two-pass engine, not minibatch EMA")
