"""Differentiable [0,1] feature extraction; parameter freezing is explicit."""
import torch
from importlib.metadata import version
from torch import nn
from torch.nn import functional as F
from recon_fd.provenance import fingerprint, state_fingerprint


class _BicubicAntialiasResize(torch.autograd.Function):
    """Preserve bicubic-AA features while isolating CUDA's non-deterministic VJP."""

    @staticmethod
    def forward(ctx, images, size):
        ctx.input_shape = tuple(images.shape)
        ctx.size = tuple(size)
        return F.interpolate(images, ctx.size, mode="bicubic",
                             align_corners=False, antialias=True)

    @staticmethod
    def backward(ctx, grad_output):
        # The resize is linear in its input. Recompute its VJP on a zero probe
        # so the sole non-deterministic CUDA kernel can run without changing
        # the forward preprocessing or relaxing other training operations.
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(False)
            with torch.enable_grad():
                probe = grad_output.new_zeros(ctx.input_shape, requires_grad=True)
                resized = F.interpolate(probe, ctx.size, mode="bicubic",
                                        align_corners=False, antialias=True)
                gradient, = torch.autograd.grad(resized, probe, grad_output)
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        return gradient, None


def _resize_bicubic_antialias(images, size):
    if images.is_cuda and images.requires_grad and torch.are_deterministic_algorithms_enabled():
        return _BicubicAntialiasResize.apply(images, size)
    return F.interpolate(images, size, mode="bicubic", align_corners=False, antialias=True)


class TinyRepresentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(12, 6)
        self.dimension = 6

    def forward(self, images):
        height, width = images.shape[-2:]
        if height % 2 == 0 and width % 2 == 0:
            # Equal nonoverlapping bins are the adaptive-2 pooling result, but
            # avg_pool2d has a deterministic CUDA backward for this tiny fixture.
            pooled = F.avg_pool2d(images, kernel_size=(height // 2, width // 2))
        else:
            pooled = F.adaptive_avg_pool2d(images, 2)
        return self.projection(pooled.flatten(1))


class InceptionRepresentation(nn.Module):
    def __init__(self, weights=""):
        super().__init__()
        from .inception import InceptionV3, INCEPTION_URL
        self.model = InceptionV3(normalize=False)
        state = (torch.load(weights, map_location="cpu", weights_only=True) if weights else
                 torch.hub.load_state_dict_from_url(INCEPTION_URL, map_location="cpu", check_hash=True))
        self.model.load_state_dict(state, strict=True)
        self.dimension = 2048

    def forward(self, images):
        return self.model(images)[0]


class TimmRepresentation(nn.Module):
    """Matches FD-Loss feature/pooling convention, NOT classification logits."""
    def __init__(self, spec):
        super().__init__()
        import timm
        from timm.data import resolve_data_config
        kwargs = {"pretrained": not bool(spec.get("weights")), "num_classes": 0}
        try:
            self.model = timm.create_model(spec["model_name"], dynamic_img_size=True,
                                          dynamic_img_pad=True, **kwargs)
        except TypeError:
            self.model = timm.create_model(spec["model_name"], **kwargs)
        if spec.get("weights"):
            self.model.load_state_dict(torch.load(spec["weights"], map_location="cpu", weights_only=True))
        config = resolve_data_config(self.model.pretrained_cfg)
        self.size = spec.get("target_size") or config["input_size"][-1]
        self.pool = spec.get("pool", "cls")
        self.prefixes = getattr(self.model, "num_prefix_tokens", 0)
        self.dimension = self.model.num_features
        self.register_buffer("mean", torch.tensor(config["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config["std"]).view(1, 3, 1, 1))

    def forward(self, images):
        if images.shape[-2:] != (self.size, self.size):
            images = _resize_bicubic_antialias(images, (self.size, self.size))
        features = self.model.forward_features((images - self.mean) / self.std)
        if features.ndim == 4:
            return features.mean((2, 3)).float()
        if self.pool == "avg":
            return features[:, self.prefixes:].mean(1).float()
        if self.prefixes:
            return features[:, 0].float()
        if getattr(self.model, "attn_pool", None) is not None:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            if pool is None:
                raise ValueError("Attention pooling API unsupported; do not silently substitute mean pooling")
            return pool(features).float()
        return features.mean(1).float()


def build_representation(spec):
    # Model initialization must not consume the tokenizer/training RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.get("seed", 173))
        kind = spec["kind"]
        if kind == "tiny":
            model = TinyRepresentation()
        elif kind == "inception":
            model = InceptionRepresentation(spec.get("weights", ""))
        elif kind == "timm":
            model = TimmRepresentation(spec)
        else:
            raise ValueError(f"Unsupported representation: {kind}")
    model.eval().requires_grad_(False)
    model.identity = {"spec": spec, "weights_sha256": state_fingerprint(model),
                      "preprocess_version": "fd-loss-5c03b811-float01-v1",
                      "torch_version": torch.__version__}
    if kind == "timm":
        model.identity["timm_version"] = version("timm")
    model.fingerprint = fingerprint(model.identity)
    return model
