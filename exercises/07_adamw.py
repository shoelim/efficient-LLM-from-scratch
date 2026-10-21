"""Week 7: implement one AdamW step in scalar arithmetic."""


def adamw_one_step(theta: float, gradient: float, m: float, v: float,
                   *, step: int, lr: float, beta1: float = 0.9,
                   beta2: float = 0.999, eps: float = 1e-8,
                   weight_decay: float = 0.01) -> tuple[float, float, float]:
    """Return (new_theta, new_m, new_v), correcting moment bias at `step`."""
    raise NotImplementedError("Do not confuse L2 penalty with decoupled decay")
