"""Contiguous next-token prediction windows, with an explicit packing convention."""

import numpy as np
import torch


def get_batch(tokens: np.ndarray, block_size: int, batch_size: int,
              generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random packed windows; EOS marks document boundaries within windows.

    All windows lie strictly within *one dataset split*. The objective predicts the
    next token even across EOS -> next-document boundary. This is deliberate packing.
    """
    if len(tokens) <= block_size or block_size < 1 or batch_size < 1:
        raise ValueError("Need more than block_size tokens and positive batch dimensions")
    # i + block_size must index a valid target, hence high = len(tokens) - block_size.
    starts = torch.randint(0, len(tokens) - block_size, (batch_size,), generator=generator)
    idx = starts.numpy()
    x = np.stack([tokens[i:i + block_size] for i in idx]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + block_size + 1] for i in idx]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
