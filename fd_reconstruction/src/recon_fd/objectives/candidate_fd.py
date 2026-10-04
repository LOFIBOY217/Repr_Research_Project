"""C and explicitly labelled ablations; no dependency on a static teacher."""
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import nn
from recon_fd.data import selected_dataset, sequential_loader
from recon_fd.representations import build_representation
from recon_fd.representations.lora import apply_qkv_lora
from recon_fd.vendor.fd_loss.queue import FeatureQueue
from .adaptive_fd import AdvStatsEMA
from .real_reference import RealReference, RealReferenceView
from .statistics import check_features
from .whitening import real_whitened_frechet_distance


@dataclass
class CandidateResult:
    fd: torch.Tensor
    real: RealReferenceView
    fake_features: torch.Tensor
    fake_version: int
    psi_version: int


class CandidateFD(nn.Module):
    def __init__(self, config, dataset, static=None):
        super().__init__()
        self.config = deepcopy(config["adaptive"])
        self.method = config["method"]
        self.static = static
        self.extractor = build_representation(self.config["representation"])
        self.scope = self.config["trainable_scope"]
        self.lora_modules = 0
        if self.scope == "full":
            self.extractor.requires_grad_(True)
        elif self.scope == "lora":
            lora = self.config["lora"]
            self.lora_modules = apply_qkv_lora(self.extractor, lora["rank"], lora["alpha"], lora["dropout"])
        else:
            raise ValueError("Candidate scope must be full or lora")
        if self.config["gradient_checkpointing"]:
            self.extractor.model.set_grad_checkpointing(True)
        self.trainable_names = tuple(n for n, p in self.extractor.named_parameters() if p.requires_grad)
        if not self.trainable_names:
            raise ValueError("Empty candidate parameter set")
        reference = self.config["real_stats"]
        pool = selected_dataset(dataset, reference["samples"], reference["seed"], True)
        self.real_reference = RealReference(self.extractor.dimension, reference,
                                            self.config["ema_beta"], pool, config["data"]["workers"])
        self.fake_statistics = AdvStatsEMA(self.extractor.dimension, self.config["ema_beta"])
        self.register_buffer("critic_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False))
        self.set_critic_trainable(False)

    @property
    def spaces(self):
        return self.static.spaces if self.static is not None else {}

    def train(self, mode=True):
        super().train(mode)
        self.extractor.eval()  # Full parameters, but no extra BN-buffer ablation.
        if self.static is not None:
            self.static.train(mode)
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
        return {"method": self.method, "scope": self.scope, "lora_modules": self.lora_modules,
                "trainable_names": list(self.trainable_names),
                "trainable_parameters": sum(p.numel() for p in self.critic_parameters()),
                "total_parameters": sum(p.numel() for p in self.extractor.parameters()),
                "static_enabled": self.static is not None, "config": self.config,
                "real_pool_identity": self.real_reference.pool.identity,
                "real_covariance_ddof": 1, "fake_statistics": "official_advfd_ema"}

    def critic_due(self, step):
        return step % self.config["update_freq"] == 0

    @torch.no_grad()
    def initialize(self, model, dataset, seed, workers):
        if self.initialized:
            raise RuntimeError("Candidate statistics already initialized")
        device = self.critic_updates.device
        self.real_reference.initialize(self.extractor, int(self.critic_updates))
        count = self.config["initialization_samples"]
        subset = selected_dataset(dataset, count, seed + 1, True)
        # Only a statistics accumulator; there is no frozen feature network.
        queue = FeatureQueue(size=count, feat_dim=self.extractor.dimension,
                             ema_beta=self.config["ema_beta"]).to(device)
        previous = model.training
        model.eval()
        try:
            for batch in sequential_loader(subset, self.config["initialization_batch_size"], workers):
                features = self.extractor(model(batch["image"].to(device)))
                check_features(features)
                queue.accumulate_batch(features)
            queue._finalize_streaming_init()
            self.fake_statistics.initialize_mean_m2(queue.mu_ema, queue.m2_ema)
        finally:
            model.train(previous)
        self.initialized.fill_(True)

    def dynamic(self, images, reconstruction, step):
        if not self.initialized:
            raise RuntimeError("Candidate statistics not initialized")
        version = int(self.critic_updates)
        real = self.real_reference.preview(self.extractor, images, version)
        features = self.extractor(reconstruction)
        fake = self.fake_statistics.preview(features)
        fd = real_whitened_frechet_distance(real.moments.mean, real.moments.cov,
                                             fake.mean, fake.cov, self.config["whiten_eps"])
        return CandidateResult(fd, real, features.detach(), int(self.fake_statistics.updates), version)

    def validate_pending(self, result):
        if result.psi_version != int(self.critic_updates):
            raise RuntimeError("Stale psi version in candidate loss")
        if result.fake_version != int(self.fake_statistics.updates):
            raise RuntimeError("Stale or duplicate candidate statistics commit")
        self.real_reference.validate_pending(result.real, int(self.critic_updates))

    @torch.no_grad()
    def commit_dynamic(self, result):
        self.validate_pending(result)
        self.real_reference.commit(result.real, int(self.critic_updates))
        self.fake_statistics.commit(result.fake_features, result.fake_version)
