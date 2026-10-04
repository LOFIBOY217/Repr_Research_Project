"""AdvFD fused-QKV LoRA convention (MIT; upstream repr_models.py).

The paper-backbone recipes target attn.qkv with Q, K and V all updated.
Do not silently enable MLP/projection/full-backbone tuning for these recipes.
"""
import torch
from torch import nn


class LoRAQKVLinear(nn.Module):
    def __init__(self, base, rank=16, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features % 3 or rank <= 0:
            raise ValueError("Expected a fused QKV Linear and positive LoRA rank")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = int(rank), float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


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
