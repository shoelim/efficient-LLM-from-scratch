"""Normalization, positional encoding, and feed-forward networks from primitives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScratchLayerNorm(nn.Module):
    """LayerNorm over the last dimension, including PyTorch's biased variance."""

    def __init__(self, width: int, *, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute statistics in fp32 even when hidden states use fp16/bf16.
        stable = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        mean = stable.mean(dim=-1, keepdim=True)
        var = ((stable - mean) ** 2).mean(dim=-1, keepdim=True)
        y = ((stable - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)
        return y * self.weight + (self.bias if self.bias is not None else 0)


class RMSNorm(nn.Module):
    def __init__(self, width: int, *, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stable = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        scale = torch.rsqrt(stable.square().mean(dim=-1, keepdim=True) + self.eps)
        return (stable * scale).to(x.dtype) * self.weight


def rotary_embeddings(x: torch.Tensor, positions: torch.Tensor,
                      base: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to (..., seq, head_dim) with even head_dim; preserves inner products.

    Here x is (B,H,T,D) and positions is (T,), with absolute positions for caches.
    """
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE needs even head dimension")
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angles = positions.float()[:, None] * frequencies[None, :]
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class FeedForward(nn.Module):
    def __init__(self, width: int, *, kind: str, dropout: float, bias: bool):
        super().__init__()
        self.kind = kind
        hidden = 4 * width if kind == "gelu" else max(1, int(8 * width / 3))
        self.fc = nn.Linear(width, hidden, bias=bias)
        self.gate = nn.Linear(width, hidden, bias=bias) if kind == "swiglu" else None
        self.proj = nn.Linear(hidden, width, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            h = F.gelu(self.fc(x), approximate="tanh")
        else:
            h = F.silu(self.gate(x)) * self.fc(x)
        return self.dropout(self.proj(h))
