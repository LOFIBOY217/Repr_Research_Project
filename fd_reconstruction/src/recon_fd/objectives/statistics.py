"""EMA/queue moments adapted from official FD-Loss, with explicit commits."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Moments:
    mean: torch.Tensor
    cov: torch.Tensor
    count: int


def check_features(features):
    if features.ndim != 2 or features.shape[0] < 1:
        raise ValueError("Expected a nonempty [batch, feature_dim] tensor")
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite features")


class RunningMoments:
    """Count-weighted Chan accumulation for finite-dataset evaluation, NOT EMA."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.scatter = None

    @torch.no_grad()
    def update(self, features):
        check_features(features)
        x = features.detach().double()
        n, mean = len(x), x.mean(0)
        centered = x - mean
        scatter = centered.T @ centered
        if self.count == 0:
            self.mean, self.scatter, self.count = mean, scatter, n
            return
        delta = mean - self.mean
        total = self.count + n
        self.scatter += scatter + torch.outer(delta, delta) * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def moments(self, ddof=1):
        if self.count <= ddof:
            raise ValueError("Insufficient observations for covariance")
        return Moments(self.mean, self.scatter / (self.count - ddof), self.count)


class EMAStats(nn.Module):
    def __init__(self, dimension, beta):
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("EMA beta must be in [0, 1)")
        self.beta = float(beta)
        self.register_buffer("mean", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer("second", torch.zeros(dimension, dimension, dtype=torch.float64))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def initialize(self, moments):
        # Convert unbiased sample covariance back to population second moment.
        if moments.count < 2:
            raise ValueError("EMA initialization requires >=2 samples")
        self.mean.copy_(moments.mean)
        self.second.copy_(moments.cov * ((moments.count - 1) / moments.count)
                          + torch.outer(moments.mean, moments.mean))
        self.initialized.fill_(True)
        self.updates.zero_()

    def preview(self, features):
        check_features(features)
        if not self.initialized:
            raise RuntimeError("Initialize EMA before training")
        x = features.double()
        mu = self.beta * self.mean.detach().clone() + (1 - self.beta) * x.mean(0)
        m2 = self.beta * self.second.detach().clone() + (1 - self.beta) * (x.T @ x / len(x))
        return Moments(mu, m2 - torch.outer(mu, mu), len(x))

    @torch.no_grad()
    def commit(self, features, expected_version):
        if int(self.updates) != expected_version:
            raise RuntimeError("Stale or duplicate statistics commit")
        moments = self.preview(features.detach())
        self.mean.copy_(moments.mean)
        self.second.copy_(moments.cov + torch.outer(moments.mean, moments.mean))
        self.updates.add_(1)


class FeatureQueue(nn.Module):
    """Full circular queue; current batch replaces the next eviction region."""
    def __init__(self, dimension, capacity):
        super().__init__()
        if capacity < 2:
            raise ValueError("Queue capacity must be >=2")
        self.capacity = capacity
        self.register_buffer("features", torch.zeros(capacity, dimension, dtype=torch.float64))
        self.register_buffer("pointer", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def initialize(self, features):
        check_features(features)
        if features.shape != self.features.shape:
            raise ValueError("Queue initialization must fill its exact capacity")
        self.features.copy_(features)
        self.pointer.zero_()
        self.updates.zero_()
        self.initialized.fill_(True)

    def _indices(self, n):
        if not self.initialized or n > self.capacity:
            raise ValueError("Queue uninitialized or batch larger than capacity")
        return (torch.arange(n, device=self.features.device) + self.pointer) % self.capacity

    def preview(self, features):
        check_features(features)
        snapshot = self.features.detach().clone()
        snapshot[self._indices(len(features))] = features.double()
        mean = snapshot.mean(0)
        centered = snapshot - mean
        return Moments(mean, centered.T @ centered / (self.capacity - 1), self.capacity)

    @torch.no_grad()
    def commit(self, features, expected_version):
        if int(self.updates) != expected_version:
            raise RuntimeError("Stale or duplicate statistics commit")
        check_features(features)
        self.features[self._indices(len(features))] = features.detach().double()
        self.pointer.copy_((self.pointer + len(features)) % self.capacity)
        self.updates.add_(1)
