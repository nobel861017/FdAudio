"""Minimal LoRA for FD post-training.

LoRALinear is a drop-in replacement for nn.Linear: the base weight/bias keep
the same attribute names so existing checkpoints load unchanged, and the
low-rank adapters (lora_A, lora_B) are extra parameters. At init the adapter
is identity (lora_B = 0), so a freshly injected model is numerically
equivalent to the original.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> adapter contributes nothing at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out

    def merged_weight(self) -> torch.Tensor:
        return self.weight + (self.lora_B @ self.lora_A) * self.scaling


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 16.0,
                targets=('qkv',)) -> int:
    """Replace nn.Linear children whose attribute name contains any target
    substring with a LoRALinear that copies the current base weights."""
    to_replace = []
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in child_name for t in targets):
                to_replace.append((module, child_name, child))
    for module, child_name, child in to_replace:
        lora = LoRALinear(child.in_features, child.out_features,
                          bias=child.bias is not None, rank=rank, alpha=alpha)
        lora.weight.data.copy_(child.weight.data)
        if child.bias is not None:
            lora.bias.data.copy_(child.bias.data)
        lora.to(device=child.weight.device, dtype=child.weight.dtype)
        setattr(module, child_name, lora)
    return len(to_replace)


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze every parameter except lora_A / lora_B."""
    for name, p in model.named_parameters():
        p.requires_grad_(('lora_A' in name) or ('lora_B' in name))


def lora_merged_state_dict(model: nn.Module) -> dict:
    """State dict with each LoRALinear merged back into a plain weight and the
    lora_A/lora_B keys removed — loadable by a vanilla (non-LoRA) model."""
    named_modules = dict(model.named_modules())
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if '.' not in k:          # top-level param (latent_mean, empty_string_feat, ...)
            out[k] = v
            continue
        parent, leaf = k.rsplit('.', 1)
        mod = named_modules.get(parent, None)
        if isinstance(mod, LoRALinear):
            if leaf in ('lora_A', 'lora_B'):
                continue
            if leaf == 'weight':
                out[k] = mod.merged_weight().detach()
            else:
                out[k] = v
        else:
            out[k] = v
    return out
