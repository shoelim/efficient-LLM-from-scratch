"""Decoder-only autoregressive Transformer. No pretrained weights or nn.Transformer."""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from llmfs.config import GPTConfig
from llmfs.model.attention import KV
from llmfs.inference.cache import StaticKVCache
from llmfs.model.block import TransformerBlock, normalization


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: tuple[KV, ...] | None = None


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = (
            nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.position == "learned" else None
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = normalization(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        # Keep deep residual branches initially small (GPT-2-style scaling).
        # Both attention output and MLP output feed residual additions.
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, 0, 0.02 / math.sqrt(2 * cfg.n_layer))
            nn.init.normal_(block.mlp.proj.weight, 0, 0.02 / math.sqrt(2 * cfg.n_layer))
        if cfg.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None,
                *, past_key_values: tuple[KV, ...] | None = None,
                use_cache: bool = False, static_cache: StaticKVCache | None = None,
                logits_last_only: bool = False) -> ModelOutput:
        if tokens.ndim != 2 or tokens.dtype not in (torch.int64, torch.int32):
            raise ValueError("tokens must be a (batch, sequence) integer tensor")
        batch, length = tokens.shape
        if length == 0:
            raise ValueError("Input sequence cannot be empty")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("Expected one cached KV pair per block")
        if static_cache is not None and targets is not None:
            raise ValueError("Static KV caching is inference-only; targets must be None")
        if static_cache is not None and past_key_values is not None:
            raise ValueError("Choose either static_cache or past_key_values")
        if static_cache is not None and static_cache.batch_size != batch:
            raise ValueError("KV cache batch size differs from tokens")
        past_len = (static_cache.length if static_cache is not None else
                    (0 if past_key_values is None else past_key_values[0][0].shape[-2]))
        if length + past_len > self.config.block_size:
            raise ValueError("Context longer than block_size: crop or re-prefill generation")
        if logits_last_only and targets is not None:
            raise ValueError("Cannot compute full cross-entropy with logits_last_only=True")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("Target tensor must have the same shape as input")
        x = self.token_embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(past_len, past_len + length, device=tokens.device)
            x = x + self.position_embedding(positions)
        x = self.dropout(x)
        updated: list[KV] = []
        for i, block in enumerate(self.blocks):
            cache = None if past_key_values is None else past_key_values[i]
            if self.gradient_checkpointing and self.training:
                if use_cache or static_cache is not None or past_key_values is not None:
                    raise ValueError("Gradient checkpointing is incompatible with inference KV caching")
                # Default argument pins the correct block for backward recomputation.
                x = checkpoint(lambda hidden, b=block: b(hidden)[0], x, use_reentrant=False)
                next_kv = None
            else:
                x, next_kv = block(x, past_kv=cache, use_cache=use_cache,
                                   static_cache=static_cache, layer_index=i)
            if next_kv is not None:
                updated.append(next_kv)
        if static_cache is not None:
            static_cache.advance(length)
        if logits_last_only:
            x = x[:, -1:, :]
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1).long(), ignore_index=-100)
        return ModelOutput(logits, loss, tuple(updated) if use_cache else None)

    def make_cache(self, batch_size: int, *, dtype: torch.dtype | None = None) -> StaticKVCache:
        """Allocate inference storage once; supply AMP activation dtype when autocasting."""
        return StaticKVCache(
            n_layers=self.config.n_layer, batch_size=batch_size,
            n_kv_heads=self.config.n_kv_head or self.config.n_head,
            max_seq_len=self.config.block_size,
            head_dim=self.config.n_embd // self.config.n_head,
            device=self.token_embedding.weight.device,
            dtype=dtype or self.token_embedding.weight.dtype)

    def count_parameters(self) -> int:
        """Count unique parameter tensors; tied embedding counted only once."""
        return sum(p.numel() for p in self.parameters())
