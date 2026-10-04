"""B baseline only: frozen static FD + upstream AdvFD, NOT our candidate."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import nn
from .statistics import Moments, check_features
from .official_fd import OfficialFDStatistics, OfficialStaticFD
from recon_fd.vendor.advfd.adversarial import FeatureStatsEMA
from .whitening import real_whitened_frechet_distance
from recon_fd.representations.lora import apply_qkv_lora


class AdvStatsEMA(FeatureStatsEMA):
    """Unmodified upstream statistics operations plus a versioned lifecycle."""
    def __init__(self, dimension, beta):
        super().__init__(dimension, beta)
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self.mu_ema

    @property
    def second(self):
        return self.m2_ema

    @torch.no_grad()
    def initialize_mean_m2(self, mean, second):
        self.initialize_from_mean_m2(mean, second)
        self.updates.zero_()

    @torch.no_grad()
    def initialize_mean_cov(self, mean, covariance):
        self.initialize_from_mean_cov(mean, covariance)
        self.updates.zero_()

    def current(self):
        mean, cov = self.current_stats()
        return Moments(mean, cov, 0)

    def preview(self, features):
        check_features(features)
        mean, cov = self.build_stats(features)
        return Moments(mean, cov, len(features))

    @torch.no_grad()
    def commit(self, features, expected_version):
        if int(self.updates) != expected_version:
            raise RuntimeError("Stale or duplicate AdvFD statistics commit")
        check_features(features)
        self.update(features)
        self.updates.add_(1)


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
        if not isinstance(static, OfficialStaticFD):
            raise TypeError("B requires the unchanged official static FD core")
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
            if (not isinstance(state, OfficialFDStatistics) or not state.initialized
                    or not state.queue.ema_stats):
                raise RuntimeError("AdvFD requires initialized matching static EMA")
            # Upstream copies the current static fake moments AT adv start.
            self.fake_statistics.initialize_mean_m2(state.queue.mu_ema, state.queue.m2_ema)

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
