"""Slow-critic C: current real pool, current minibatch fake, no fake EMA.

This is a stochastic minibatch FD objective, not the exact 50k fake-pool FD
gradient in ``ours_current_both``. The distinction is explicit in the logs.
"""
import torch
from recon_fd.objectives.statistics import Moments, check_features
from .candidate_fd import CandidateFD, CandidateResult
from .whitening import real_whitened_frechet_distance


class CurrentBatchFD(CandidateFD):
    def build_fake_statistics(self, dimension):
        return None

    def parameter_manifest(self):
        manifest = super().parameter_manifest()
        manifest.update(fake_statistics="current_minibatch_no_ema",
                        fake_covariance_ddof=1, gradient_estimator="current_minibatch",
                        representation_update_interval_images=(
                            self.config["update_freq"] * self.config["initialization_batch_size"]))
        return manifest

    @torch.no_grad()
    def initialize(self, model, dataset, seed, workers):
        if self.initialized:
            raise RuntimeError("Current-batch statistics already initialized")
        self.real_reference.initialize(self.extractor, int(self.critic_updates))
        self.initialized.fill_(True)

    def dynamic(self, images, reconstruction, step):
        if not self.initialized:
            raise RuntimeError("Current-batch real statistics not initialized")
        version = int(self.critic_updates)
        real = self.real_reference.preview(self.extractor, None, version)
        features = self.extractor(reconstruction)
        check_features(features)
        if len(features) < 2:
            raise ValueError("Current-batch FD needs at least two reconstructions")
        mean = features.mean(0)
        centered = features - mean
        covariance = centered.T @ centered / (len(features) - 1)
        fake = Moments(mean, covariance, len(features))
        fd = real_whitened_frechet_distance(real.moments.mean, real.moments.cov,
                                            fake.mean, fake.cov, self.config["whiten_eps"])
        return CandidateResult(fd, real, features.detach(), 0, version)

    def validate_pending(self, result):
        if result.psi_version != int(self.critic_updates) or result.fake_version != 0:
            raise RuntimeError("Stale current-batch FD result")
        self.real_reference.validate_pending(result.real, int(self.critic_updates))

    @torch.no_grad()
    def commit_dynamic(self, result):
        self.validate_pending(result)
        self.real_reference.commit(result.real, int(self.critic_updates))
