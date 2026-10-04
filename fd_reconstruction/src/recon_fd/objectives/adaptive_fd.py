"""B baseline only: frozen static FD + upstream AdvFD, NOT our candidate."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import nn
from .statistics import EMAStats, Moments, check_features
from .whitening import real_whitened_frechet_distance
from recon_fd.representations.lora import apply_qkv_lora


class AdvStatsEMA(EMAStats):
    """Upstream FeatureStatsEMA initialization, with explicit versioned commit.

    Real reference covariance is copied literally, with NO ddof rescaling.
    The baseline FD-only EMA's sample-based initialization is intentionally not
    used here: AdvFD initialize_from_mean_cov has different semantics.
    """
    @torch.no_grad()
    def initialize_mean_m2(self, mean, second):
        self.mean.copy_(mean.detach().double())
        self.second.copy_(second.detach().double())
        self.initialized.fill_(True)
        self.updates.zero_()

    @torch.no_grad()
    def initialize_mean_cov(self, mean, covariance):
        self.initialize_mean_m2(mean, covariance + torch.outer(mean, mean))

    def current(self):
        if not self.initialized:
            raise RuntimeError("AdvFD EMA not initialized")
        mean = self.mean.detach().clone()
        cov = self.second.detach().clone() - torch.outer(mean, mean)
        return Moments(mean, 0.5 * (cov + cov.T), 0)

    def preview(self, features):
        if self.initialized:
            return super().preview(features)
        check_features(features)
        x = features.double()
        mean = x.mean(0)
        return Moments(mean, x.T @ x / len(x) - torch.outer(mean, mean), len(x))

    @torch.no_grad()
    def commit(self, features, expected_version):
        super().commit(features, expected_version)
        self.initialized.fill_(True)


@dataclass
class AdaptiveResult:
    fd: torch.Tensor
    real_features: torch.Tensor | None
    fake_features: torch.Tensor
    real_version: int
    fake_version: int


class AdvFD(nn.Module):
    def __init__(self, static, config):
        super().__init__()
        self.static = static
        self.config = deepcopy(config)
        self.source_name = config["representation"]
        source = static.spaces[self.source_name]
        self.extractor = deepcopy(source.extractor)
        spec = source.extractor.identity["spec"]
        if spec["kind"] in {"inception", "tiny"}:
            self.extractor.requires_grad_(True)
            self.scope = "full" if spec["kind"] == "inception" else "full_engineering"
            self.lora_modules = 0
        else:
            lora = config["lora"]
            self.lora_modules = apply_qkv_lora(self.extractor, lora["rank"], lora["alpha"], lora["dropout"])
            self.scope = "lora"
            if config["gradient_checkpointing"]:
                self.extractor.model.set_grad_checkpointing(True)
        self.trainable_names = tuple(name for name, p in self.extractor.named_parameters() if p.requires_grad)
        if not self.trainable_names:
            raise ValueError("Empty adversarial parameter set")
        self.real_statistics = AdvStatsEMA(self.extractor.dimension, config["ema_beta"])
        self.fake_statistics = AdvStatsEMA(self.extractor.dimension, config["ema_beta"])
        self.real_statistics.initialize_mean_cov(source.reference_mean, source.reference_cov)
        self.register_buffer("critic_updates", torch.tensor(0, dtype=torch.long))
        self.set_critic_trainable(False)

    @property
    def spaces(self):
        return self.static.spaces

    def train(self, mode=True):
        super().train(mode)
        self.static.train(mode)
        self.extractor.eval()  # Keep BN buffers fixed, NOT BN affine parameters.
        return self

    def set_critic_trainable(self, enabled):
        selected = set(self.trainable_names)
        for name, parameter in self.extractor.named_parameters():
            parameter.requires_grad_(enabled and name in selected)
        self.extractor.eval()

    def critic_parameters(self):
        selected = set(self.trainable_names)
        return [p for name, p in self.extractor.named_parameters() if name in selected]

    def parameter_manifest(self):
        selected = set(self.trainable_names)
        return {"representation": self.source_name, "scope": self.scope,
                "lora_modules": self.lora_modules, "trainable_names": list(self.trainable_names),
                "trainable_parameters": sum(p.numel() for name, p in self.extractor.named_parameters() if name in selected),
                "total_parameters": sum(p.numel() for p in self.extractor.parameters()),
                "config": self.config}

    def active(self, step):
        return step >= self.config["start_step"]

    def effective_weight(self, step):
        if not self.active(step):
            return 0.0
        warmup = self.config["warmup_steps"]
        fraction = min(1.0, (step - self.config["start_step"]) / warmup) if warmup else 1.0
        return self.config["weight"] * fraction

    def critic_due(self, step):
        return self.active(step) and (step - self.config["start_step"]) % self.config["update_freq"] == 0

    @torch.no_grad()
    def initialize_fake_at_activation(self):
        if not self.fake_statistics.initialized:
            state = self.spaces[self.source_name].statistics
            if not isinstance(state, EMAStats) or not state.initialized:
                raise RuntimeError("AdvFD requires initialized matching static EMA")
            # Upstream copies the current static fake moments AT adv start.
            self.fake_statistics.initialize_mean_m2(state.mean, state.second)

    def dynamic(self, real_images, fake_images, step):
        if not self.fake_statistics.initialized:
            raise RuntimeError("Initialize adversarial fake EMA before the active step")
        real_features = None
        if step % self.config["real_stats"]["update_freq"] == 0:
            with torch.no_grad():  # Paper's detach_real configuration.
                real_features = self.extractor(real_images.detach())
            real = self.real_statistics.preview(real_features)
        else:
            real = self.real_statistics.current()
        fake_features = self.extractor(fake_images)
        fake = self.fake_statistics.preview(fake_features)
        value = real_whitened_frechet_distance(real.mean, real.cov, fake.mean, fake.cov,
                                             self.config["whiten_eps"])
        return AdaptiveResult(value, real_features, fake_features.detach(),
                              int(self.real_statistics.updates), int(self.fake_statistics.updates))

    @torch.no_grad()
    def validate_pending(self, result):
        if (int(self.real_statistics.updates) != result.real_version
                or int(self.fake_statistics.updates) != result.fake_version):
            raise RuntimeError("Duplicate or stale dynamic statistics commit")

    @torch.no_grad()
    def commit_dynamic(self, result):
        self.validate_pending(result)
        if result.real_features is not None:
            self.real_statistics.commit(result.real_features, result.real_version)
        self.fake_statistics.commit(result.fake_features, result.fake_version)

    def forward(self, reconstruction):
        return self.static(reconstruction)

    def commit(self, result):
        self.static.commit(result)
