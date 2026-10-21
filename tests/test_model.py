import math

import pytest
import torch
import torch.nn.functional as F

from llmfs.config import GPTConfig
from llmfs.model.gpt import GPT
from llmfs.model.layers import RMSNorm, ScratchLayerNorm, rotary_embeddings


def cfg(**overrides):
    data = dict(vocab_size=41, block_size=16, n_layer=2, n_head=4, n_embd=32,
                dropout=0.0)
    data.update(overrides)
    return GPTConfig(**data)


def test_invalid_configs():
    for kw in (dict(n_embd=13), dict(n_kv_head=3), dict(n_kv_head=0), dict(dropout=1),
               dict(position="broken"), dict(attention_impl="wrong"),
               dict(position="rope", n_embd=20, n_head=4)):
        with pytest.raises(ValueError):
            cfg(**kw)


def test_norm_and_rotary():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 32)
    layer = ScratchLayerNorm(32)
    ref = torch.nn.LayerNorm(32)
    assert torch.allclose(layer(x), ref(x), atol=1e-6)
    y = RMSNorm(32)(x)
    assert y.shape == x.shape
    vectors = torch.randn(2, 4, 3, 8)
    rotated = rotary_embeddings(vectors, torch.tensor([0, 3, 5]))
    assert torch.allclose(rotated.square().sum(-1), vectors.square().sum(-1), atol=1e-5)
    assert torch.allclose(rotated[:, :, 0], vectors[:, :, 0])


@pytest.mark.parametrize("variant", [
    {}, {"position": "rope"},
    {"position": "rope", "norm": "rmsnorm", "mlp": "swiglu", "n_kv_head": 2},
    {"attention_impl": "sdpa"},
])
def test_model_forward_backward_and_roundtrip(variant):
    torch.manual_seed(4)
    model = GPT(cfg(**variant))
    x = torch.randint(0, 41, (2, 10))
    target = torch.randint(0, 41, (2, 10))
    out = model(x, target)
    assert out.logits.shape == (2, 10, 41)
    assert out.loss.isfinite()
    out.loss.backward()
    assert model.token_embedding.weight.grad is not None
    restored = GPT(cfg(**variant))
    restored.load_state_dict(model.state_dict())
    model.eval()
    restored.eval()
    assert torch.allclose(model(x).logits, restored(x).logits, atol=1e-6)
    with pytest.raises(ValueError, match="longer"):
        model(torch.randint(0, 41, (1, 17)))


def test_weights_are_tied_and_parameter_count_correct():
    model = GPT(cfg())
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
    untied = GPT(cfg(tie_weights=False))
    assert untied.count_parameters() - model.count_parameters() == 41 * 32


def test_causality_future_token_change_cannot_affect_past():
    torch.manual_seed(12)
    model = GPT(cfg()).eval()
    x = torch.randint(0, 41, (1, 12))
    y = x.clone()
    y[:, 7:] = torch.randint(0, 41, (1, 5))
    assert torch.allclose(model(x).logits[:, :7], model(y).logits[:, :7], atol=1e-6)


def test_manual_sdpa_parity_forward_and_gradients():
    torch.manual_seed(32)
    a = GPT(cfg(attention_impl="manual"))
    b = GPT(cfg(attention_impl="sdpa"))
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 41, (2, 11))
    targets = torch.randint(0, 41, (2, 11))
    loss_a = a(x, targets).loss
    loss_b = b(x, targets).loss
    assert torch.allclose(loss_a, loss_b, atol=2e-6)
    loss_a.backward()
    loss_b.backward()
    for (_, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=3e-6)


@pytest.mark.parametrize("variant", [
    {}, {"position": "rope"},
    {"position": "rope", "n_kv_head": 2},
    {"position": "rope", "n_kv_head": 1, "attention_impl": "sdpa"},
])
def test_cache_logit_equivalence(variant):
    torch.manual_seed(123)
    model = GPT(cfg(**variant)).eval()
    tokens = torch.randint(0, 41, (2, 12))
    full = model(tokens).logits
    cache = None
    pieces = []
    for pos in range(12):
        out = model(tokens[:, pos:pos + 1], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        pieces.append(out.logits)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=2e-5)
    assert cache[0][0].shape[:2] == (2, cfg(**variant).n_kv_head or 4)


def test_uniform_cross_entropy_and_masking():
    logits = torch.zeros(2, 3, 11)
    labels = torch.tensor([[4, 5, -100], [2, 6, -100]])
    loss = F.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1), ignore_index=-100)
    assert math.isclose(loss.item(), math.log(11), abs_tol=1e-6)
