"""Verbatim LoRAQKVLinear from AdvFD repr_models.py (4e4cfed, MIT).

Only imports and file packaging differ; keep the class source unchanged.
"""
import torch


class LoRAQKVLinear(torch.nn.Module):
    """LoRA update for fused timm QKV layers, optionally leaving K frozen."""

    def __init__(
        self,
        base: torch.nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        update_k: bool = True,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        if base.out_features % 3 != 0:
            raise ValueError("LoRAQKVLinear requires out_features divisible by 3")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.update_k = bool(update_k)
        self.head_dim = base.out_features // 3
        out_features = base.out_features if self.update_k else 2 * self.head_dim
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.lora_A = torch.nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = torch.nn.Linear(self.rank, out_features, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)
        for p in self.lora_A.parameters():
            p._fd_adv_trainable = True
        for p in self.lora_B.parameters():
            p._fd_adv_trainable = True

    def forward(self, x: torch.Tensor):
        base_out = self.base(x)
        low_rank = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        if self.update_k:
            return base_out + low_rank
        q_update, v_update = low_rank.split(self.head_dim, dim=-1)
        update = torch.zeros_like(base_out)
        update[..., : self.head_dim] = q_update
        update[..., 2 * self.head_dim :] = v_update
        return base_out + update
