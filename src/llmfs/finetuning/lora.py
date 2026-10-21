"""LoRA adapters: frozen W + (alpha/r) * B @ A, with merge/unmerge checks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("Need rank >= 1 and alpha > 0")
        self.base = linear
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.merged = False

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.merged:
            y = y + F.linear(F.linear(x, self.A), self.B) * self.scale
        return y

    @torch.no_grad()
    def merge(self) -> None:
        if not self.merged:
            self.base.weight.add_(self.B @ self.A, alpha=self.scale)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight.add_(self.B @ self.A, alpha=-self.scale)
            self.merged = False


def apply_lora(model: nn.Module, *, rank: int = 4, alpha: float = 8.0,
               targets: tuple[str, ...] = ("q_proj", "v_proj", "qkv_proj")) -> list[str]:
    """Freeze base model and replace linear submodules whose last name is in targets."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or name.rsplit(".", 1)[-1] not in targets:
            continue
        parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        adapter = LoRALinear(module, rank=rank, alpha=alpha)
        adapter.to(device=module.weight.device, dtype=module.weight.dtype)
        setattr(parent, attr, adapter)
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No linear modules matched targets: {targets}")
    return replaced


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.merged:
                raise ValueError("Unmerge LoRA modules before saving adapter-only weights")
            result[name + ".A"] = module.A.detach().cpu().clone()
            result[name + ".B"] = module.B.detach().cpu().clone()
    return result


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = set(adapter_state_dict(model))
    if set(state) != expected:
        raise ValueError(f"Adapter keys do not match: missing={expected - set(state)}, "
                         f"unexpected={set(state) - expected}")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.A.copy_(state[name + ".A"].to(module.A.device))
                module.B.copy_(state[name + ".B"].to(module.B.device))
