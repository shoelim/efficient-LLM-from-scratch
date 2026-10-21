"""Week 5: implement transparent causal self-attention without nn.MultiheadAttention."""

import torch


def causal_attention(q: torch.Tensor, k: torch.Tensor,
                     v: torch.Tensor) -> torch.Tensor:
    """Inputs shape (B,H,T,D); return (B,H,T,D).

    Divide by sqrt(D), mask future keys BEFORE softmax, then multiply V.
    Test with an exact hand-computed 2-token example.
    """
    raise NotImplementedError("Compare against model/attention.py after attempting")
