"""Held-out cross-entropy/perplexity and deterministic evaluation sampling."""

import math

import numpy as np
import torch

from llmfs.data.fast_batcher import TokenBatcher


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, tokens: np.ndarray, *, block_size: int,
                  batch_size: int, batches: int, device: torch.device,
                  seed: int = 2026, precision: str = "fp32") -> float:
    if batches < 1:
        raise ValueError("Must evaluate at least one batch")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batcher = TokenBatcher(tokens, block_size=block_size, batch_size=batch_size,
                           device=device, storage="cpu")
    losses = []
    try:
        for _ in range(batches):
            x, y = batcher.next(generator)
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
                                enabled=precision != "fp32"):
                out = model(x, y)
            losses.append(float(out.loss.item()))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def perplexity(nll: float) -> float:
    return math.exp(nll)
