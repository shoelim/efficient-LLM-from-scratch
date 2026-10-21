"""Minimal decoupled AdamW with bias correction, written explicitly for study."""

import torch
from torch.optim import Optimizer


class ScratchAdamW(Optimizer):
    """AdamW: separate weight decay from the adaptive gradient update.

    Supports dense gradients and float parameters. Optimizer states and gradients
    stay in parameter dtype; mixed-precision autocast typically retains fp32 params.
    """

    def __init__(self, params, lr: float = 3e-4, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        if not lr > 0 or not eps > 0 or weight_decay < 0:
            raise ValueError("Invalid learning rate, epsilon or weight decay")
        if any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must be in [0,1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = (group["lr"], group["betas"],
                                      group["eps"], group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                if wd:
                    parameter.mul_(1 - lr * wd)
                m.mul_(b1).add_(gradient, alpha=1 - b1)
                v.mul_(b2).addcmul_(gradient, gradient, value=1 - b2)
                corrected_m = m / (1 - b1 ** step)
                corrected_v = v / (1 - b2 ** step)
                parameter.addcdiv_(corrected_m, corrected_v.sqrt().add(eps), value=-lr)
        return loss


def create_optimizer(model: torch.nn.Module, learning_rate: float,
                     weight_decay: float, *, implementation: str = "scratch",
                     device_type: str = "cpu") -> torch.optim.Optimizer:
    """Decay matrix weights (including embeddings) but not norm scales or biases.

    Tied embeddings are encountered once via named_parameters() deduplication.
    """
    decay, no_decay = [], []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if implementation == "auto":
        implementation = "fused" if device_type == "cuda" else "torch"
    if implementation == "scratch":
        return ScratchAdamW(groups, lr=learning_rate)
    if implementation == "torch":
        return torch.optim.AdamW(groups, lr=learning_rate)
    if implementation == "fused":
        if device_type != "cuda":
            raise ValueError("Fused AdamW is intended for CUDA in this project")
        return torch.optim.AdamW(groups, lr=learning_rate, fused=True)
    raise ValueError(f"Unknown optimizer: {implementation}")
