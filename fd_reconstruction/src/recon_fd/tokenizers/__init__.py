"""Grounded-style reconstruction adapters, independent of FD representations."""
import torch
from torch import nn


class TinyReconstructor(nn.Module):
    """Offline engineering fixture; never a pretrained scientific baseline."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Tanh())
        self.decoder = nn.Sequential(nn.Conv2d(8, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, images):
        return self.decoder(self.encoder(images))

    def parameter_groups(self):
        return {"encoder": list(self.encoder.parameters()), "decoder": list(self.decoder.parameters())}


class KLReconstructor(nn.Module):
    """AutoencoderKL posterior-mean path used by Grounded SD/VA/REPA-E VAEs.

    Public input/output: [0,1]. No frozen encoder, sampling, diffusion-model
    latent scaling, or generation-time latent input. Original clamp retained.
    """
    def __init__(self, checkpoint, local_files_only=True, gradient_checkpointing=False):
        super().__init__()
        from diffusers import AutoencoderKL
        if not checkpoint:
            raise ValueError("A pretrained AutoencoderKL directory or model ID is required")
        self.model = AutoencoderKL.from_pretrained(checkpoint, local_files_only=local_files_only)
        self.model.requires_grad_(True)
        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

    def forward(self, images):
        latent = self.model.encode(images * 2 - 1).latent_dist.mean
        return (self.model.decode(latent).sample.clamp(-1, 1) + 1) * 0.5

    def parameter_groups(self):
        decoder, encoder = [], []
        for name, parameter in self.model.named_parameters():
            (decoder if name.startswith(("decoder.", "post_quant_conv.")) else encoder).append(parameter)
        return {"encoder": encoder, "decoder": decoder}


def build_tokenizer(spec):
    if spec["kind"] == "tiny":
        model = TinyReconstructor()
    elif spec["kind"] == "autoencoder_kl":
        model = KLReconstructor(spec["checkpoint"], spec.get("local_files_only", True),
                                spec.get("gradient_checkpointing", False))
    else:
        raise ValueError(f"Unsupported tokenizer kind: {spec['kind']}")
    model.requires_grad_(True)
    return model
