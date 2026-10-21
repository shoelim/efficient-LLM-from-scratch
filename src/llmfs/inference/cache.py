"""Fixed-capacity per-layer KV cache for efficient decode.

Unlike a growing ``torch.cat`` cache, storage is allocated exactly once. Each
attention layer writes to [start:end], and the model advances ``length`` only
after all layers have completed. This is a *contiguous context* cache; at the
context limit callers must stop or explicitly re-prefill a cropped prompt.
"""

import torch


class StaticKVCache:
    def __init__(self, *, n_layers: int, batch_size: int, n_kv_heads: int,
                 max_seq_len: int, head_dim: int, device: torch.device,
                 dtype: torch.dtype = torch.float32):
        if min(n_layers, batch_size, n_kv_heads, max_seq_len, head_dim) <= 0:
            raise ValueError("KV cache dimensions must be positive")
        shape = (batch_size, n_kv_heads, max_seq_len, head_dim)
        self.keys = [torch.empty(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.length = 0
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k.shape != v.shape or k.ndim != 4:
            raise ValueError("K and V must both have shape (B, Hkv, T, D)")
        start = self.length
        end = start + k.shape[-2]
        dest = self.keys[layer]
        if end > self.max_seq_len:
            raise ValueError("KV cache capacity exceeded; re-prefill a cropped context")
        if (k.shape[0] != self.batch_size or k.shape[1] != dest.shape[1]
                or k.shape[-1] != dest.shape[-1] or k.device != dest.device
                or k.dtype != dest.dtype):
            raise ValueError("KV cache tensor shape, device, or dtype mismatch")
        dest[:, :, start:end, :].copy_(k)
        self.values[layer][:, :, start:end, :].copy_(v)
        return dest[:, :, :end, :], self.values[layer][:, :, :end, :]

    def advance(self, n_tokens: int) -> None:
        if n_tokens < 0 or self.length + n_tokens > self.max_seq_len:
            raise ValueError("Invalid KV cache advance")
        self.length += n_tokens

    def reset(self) -> None:
        """Reuse storage for an independent prompt without leaking previous tokens."""
        self.length = 0
