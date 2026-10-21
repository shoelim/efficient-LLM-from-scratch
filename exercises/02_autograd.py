"""Week 2: derive the softmax-cross-entropy gradient in tensor notation."""

import torch


def cross_entropy_gradient(logits: torch.Tensor, target: int) -> torch.Tensor:
    """Compute d[-log softmax(logits)[target]] / d logits WITHOUT autograd.

    Input is a 1-D tensor; output has the same shape. Check with autograd
    and torch.autograd.gradcheck using double precision.
    """
    raise NotImplementedError("Derive p_j - indicator(j==target)")
