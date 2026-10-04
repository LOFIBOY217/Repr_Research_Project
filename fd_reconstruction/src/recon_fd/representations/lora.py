"""AdvFD fused-QKV LoRA convention (MIT; upstream repr_models.py).

The paper-backbone recipes target attn.qkv with Q, K and V all updated.
Do not silently enable MLP/projection/full-backbone tuning for these recipes.
"""
from torch import nn
from recon_fd.vendor.advfd.lora import LoRAQKVLinear


def apply_qkv_lora(representation, rank=16, alpha=16.0, dropout=0.0):
    model = representation.model
    replacements = [(name, module) for name, module in model.named_modules()
                    if name.endswith(".attn.qkv") and isinstance(module, nn.Linear)]
    if not replacements:
        raise ValueError("No attn.qkv Linear modules found; unsupported LoRA backbone")
    representation.requires_grad_(False)
    for name, module in replacements:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRAQKVLinear(module, rank, alpha, dropout))
    return len(replacements)
