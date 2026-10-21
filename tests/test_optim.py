import math
from copy import deepcopy

import pytest
import torch

from llmfs.training.optim import ScratchAdamW
from llmfs.training.schedule import cosine_lr


def test_adamw_matches_pytorch_for_multiple_steps():
    p1 = torch.nn.Parameter(torch.tensor([0.3, -0.2, 0.9], dtype=torch.double))
    p2 = torch.nn.Parameter(p1.detach().clone())
    a = ScratchAdamW([p1], lr=0.05, weight_decay=0.2)
    b = torch.optim.AdamW([p2], lr=0.05, weight_decay=0.2)
    for step in range(9):
        grad = torch.tensor([0.2 + step * 0.1, -0.4, 0.03], dtype=torch.double)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        a.step()
        b.step()
        assert torch.allclose(p1, p2, atol=1e-10)


def test_adamw_state_roundtrip():
    x = torch.nn.Parameter(torch.ones(3))
    opt = ScratchAdamW([x])
    x.grad = torch.tensor([2., 1., 3.])
    opt.step()
    state = deepcopy(opt.state_dict())
    y = torch.nn.Parameter(x.detach().clone())
    opt2 = ScratchAdamW([y])
    opt2.load_state_dict(state)
    x.grad = y.grad = torch.tensor([1., 2., 3.])
    opt.step()
    opt2.step()
    assert torch.equal(x, y)


def test_scheduler_warmup_and_final_decay():
    values = [cosine_lr(i, 10, 2, 0.01, 0.001) for i in range(10)]
    assert values[0] == 0.005
    assert values[1] == 0.01
    assert math.isclose(values[-1], 0.001)
    assert all(values[i] >= values[i + 1] for i in range(1, 9))
    with pytest.raises(ValueError):
        cosine_lr(10, 10, 2, 0.01, 0.001)


def test_gradient_accumulation_agrees_with_full_batch():
    torch.manual_seed(0)
    base = torch.nn.Linear(5, 3)
    full = torch.nn.Linear(5, 3)
    full.load_state_dict(base.state_dict())
    x = torch.randn(8, 5)
    y = torch.randn(8, 3)
    ((full(x) - y)**2).mean().backward()
    for i in (0, 4):
        (((base(x[i:i+4]) - y[i:i+4])**2).mean() / 2).backward()
    for p, q in zip(base.parameters(), full.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-6)
