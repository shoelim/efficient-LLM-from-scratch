"""Stateless warmup+cosine decay; the step index is zero-based."""

import math


def cosine_lr(step: int, total_steps: int, warmup_steps: int, max_lr: float,
              min_lr: float) -> float:
    if not 0 <= step < total_steps:
        raise ValueError("Step must be in [0, total_steps)")
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return min_lr if step == total_steps - 1 else max_lr
    fraction = (step - warmup_steps) / (decay_steps - 1)
    if step == warmup_steps:
        return max_lr
    if step == total_steps - 1:
        return min_lr
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * fraction))
