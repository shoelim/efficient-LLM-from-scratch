"""Week 10: write a top-k categorical sampler from logits."""

import torch


def sample_top_k(logits: torch.Tensor, k: int,
                 generator: torch.Generator) -> int:
    """Take a 1-D vocabulary logits vector, return a sampled integer ID.

    Numerical hint: subtract max logits before exponentiating, or use softmax.
    """
    raise NotImplementedError("Derive masked distribution before writing code")
