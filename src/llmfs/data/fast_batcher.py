"""Vectorized contiguous LM window sampling, with optional GPU-resident tokens.

A CPU-backed token array (typically a memmap) remains on host; only requested
windows are copied to the accelerator. ``batch_device=device`` instead copies
all token IDs to the training device once, trading VRAM for faster sampling.
The random-start generator is always on CPU to allow exact RNG-state restore.
"""

import numpy as np
import torch


class TokenBatcher:
    def __init__(self, tokens: np.ndarray, *, block_size: int, batch_size: int,
                 device: torch.device, storage: str = "cpu"):
        if tokens.ndim != 1 or len(tokens) <= block_size:
            raise ValueError("Dataset needs > block_size tokens")
        if block_size < 1 or batch_size < 1 or storage not in ("cpu", "device"):
            raise ValueError("Invalid batcher dimensions/storage")
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.storage = storage
        self.length = len(tokens)
        self.offset = np.arange(block_size + 1, dtype=np.int64)
        self.tokens_cpu = tokens if storage == "cpu" else None
        if storage == "device":
            # Copy uint32 IDs via int32 for half the token-table memory; IDs should fit signed int32.
            self.tokens_gpu = torch.from_numpy(np.asarray(tokens, dtype=np.int32)).to(device)
            self.offset_gpu = torch.arange(block_size + 1, dtype=torch.long, device=device)
        else:
            self.tokens_gpu = None
            self.offset_gpu = None

    def next(self, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.length - self.block_size,
                               (self.batch_size,), generator=rng, device="cpu")
        if self.tokens_gpu is not None:
            windows = self.tokens_gpu[starts.to(self.device)[:, None] + self.offset_gpu[None, :]]
        else:
            rows = self.tokens_cpu[starts.numpy()[:, None] + self.offset[None, :]]
            # np.memmap's uint32 is converted to supported PyTorch index dtype.
            windows = torch.from_numpy(rows.astype(np.int64, copy=False))
            if self.device.type == "cuda":
                windows = windows.pin_memory().to(self.device, non_blocking=True)
            else:
                windows = windows.to(self.device)
        return windows[:, :-1], windows[:, 1:]
