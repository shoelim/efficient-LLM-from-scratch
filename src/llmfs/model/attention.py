"""Readable reference attention + optimized SDPA, GQA, fused QKV and static KV cache.

SDPA's is_causal=True avoids materializing the attention mask on prefill and
training. For single-token cached decode, all existing keys are visible and
is_causal=False is correct. Multi-token cached chunks need an offset mask.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmfs.config import GPTConfig
from llmfs.inference.cache import StaticKVCache

KV = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_head if cfg.n_kv_head is None else cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.position = cfg.position
        self.rope_base = cfg.rope_base
        self.impl = cfg.attention_impl
        self.fused_qkv = cfg.fused_qkv
        self.dropout_probability = cfg.dropout
        kv_width = self.n_kv_head * self.head_dim
        if cfg.fused_qkv:
            self.qkv_proj = nn.Linear(cfg.n_embd, cfg.n_embd + 2 * kv_width, bias=cfg.bias)
        else:
            self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
            self.k_proj = nn.Linear(cfg.n_embd, kv_width, bias=cfg.bias)
            self.v_proj = nn.Linear(cfg.n_embd, kv_width, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        if cfg.position == "rope":
            # Nonpersistent buffers rebuild from config and move with module.to(device).
            positions = torch.arange(cfg.block_size, dtype=torch.float32)[:, None]
            freqs = cfg.rope_base ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
            angles = positions * freqs[None, :]
            self.register_buffer("_rope_cos", angles.cos(), persistent=False)
            self.register_buffer("_rope_sin", angles.sin(), persistent=False)

    def _rope(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """RoPE in float32 for phase accuracy, then cast to activation dtype."""
        length = x.size(-2)
        cos = self._rope_cos[start:start + length][None, None].to(dtype=x.dtype)
        sin = self._rope_sin[start:start + length][None, None].to(dtype=x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, *, past_kv: KV | None = None,
                use_cache: bool = False, static_cache: StaticKVCache | None = None,
                layer_index: int = 0) -> tuple[torch.Tensor, KV | None]:
        batch, length, width = x.shape
        if static_cache is not None and past_kv is not None:
            raise ValueError("Provide either a static or dynamic KV cache, not both")
        past_len = (static_cache.length if static_cache is not None else
                    (0 if past_kv is None else past_kv[0].shape[-2]))
        if self.fused_qkv:
            q, k, v = self.qkv_proj(x).split((width, self.n_kv_head * self.head_dim,
                                             self.n_kv_head * self.head_dim), dim=-1)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.position == "rope":
            q = self._rope(q, past_len)
            k = self._rope(k, past_len)
        if static_cache is not None:
            k, v = static_cache.append(layer_index, k, v)
        elif past_kv is not None:
            old_k, old_v = past_kv
            if old_k.shape[:2] != (batch, self.n_kv_head) or old_v.shape != old_k.shape:
                raise ValueError("Incorrect past KV dimensions")
            k, v = torch.cat((old_k, k), dim=-2), torch.cat((old_v, v), dim=-2)
        current_cache = (k, v) if use_cache and static_cache is None else None
        total_len = k.shape[-2]
        gqa = self.n_head != self.n_kv_head
        # SDPA supports native GQA on CUDA; other backends use the portable expansion.
        native_gqa = gqa and self.impl == "sdpa" and x.device.type == "cuda"
        if gqa and not native_gqa:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        if self.impl == "sdpa":
            if past_len == 0:
                allowed, is_causal = None, True
            elif length == 1:
                allowed, is_causal = None, False
            else:
                query_pos = torch.arange(past_len, past_len + length, device=x.device)[:, None]
                key_pos = torch.arange(total_len, device=x.device)[None, :]
                allowed = (key_pos <= query_pos)[None, None, :, :]
                is_causal = False
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed,
                dropout_p=self.dropout_probability if self.training else 0.0,
                is_causal=is_causal, enable_gqa=native_gqa)
        else:
            query_pos = torch.arange(past_len, past_len + length, device=x.device)[:, None]
            key_pos = torch.arange(total_len, device=x.device)[None, :]
            allowed = (key_pos <= query_pos)[None, None, :, :]
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probabilities = self.attn_dropout(F.softmax(scores.float(), dim=-1).to(q.dtype))
            y = probabilities @ v
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.out_proj(y)), current_cache
