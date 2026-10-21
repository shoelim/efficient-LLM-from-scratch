"""Pre-normalization Transformer block."""

import torch
import torch.nn as nn

from llmfs.config import GPTConfig
from llmfs.model.attention import KV, CausalSelfAttention
from llmfs.inference.cache import StaticKVCache
from llmfs.model.layers import FeedForward, RMSNorm


def normalization(cfg: GPTConfig) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return nn.RMSNorm(cfg.n_embd, eps=1e-5) if hasattr(nn, "RMSNorm") else RMSNorm(cfg.n_embd)
    return nn.LayerNorm(cfg.n_embd, eps=1e-5, elementwise_affine=True, bias=cfg.bias)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = normalization(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = normalization(cfg)
        self.mlp = FeedForward(cfg.n_embd, kind=cfg.mlp, dropout=cfg.dropout, bias=cfg.bias)

    def forward(self, x: torch.Tensor, *, past_kv: KV | None = None,
                use_cache: bool = False, static_cache: StaticKVCache | None = None,
                layer_index: int = 0) -> tuple[torch.Tensor, KV | None]:
        attn, new_cache = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache,
                                        static_cache=static_cache, layer_index=layer_index)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, new_cache
