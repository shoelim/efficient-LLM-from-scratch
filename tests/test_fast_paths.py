"""Cross-implementation tests target the subtle correctness hazards in fast paths."""

import numpy as np
import pytest
import torch

from llmfs.config import GPTConfig
from llmfs.data.fast_batcher import TokenBatcher
from llmfs.inference.sample import generate
from llmfs.model.gpt import GPT


def make_cfg(**updates):
    params = dict(vocab_size=37, block_size=20, n_layer=2, n_head=4,
                  n_embd=32, dropout=0, bias=False, position="rope",
                  n_kv_head=2, attention_impl="sdpa", fused_qkv=True,
                  norm="rmsnorm", mlp="swiglu")
    params.update(updates)
    return GPTConfig(**params)


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
@pytest.mark.parametrize("position", ["rope", "learned"])
def test_static_prefill_decode_and_multitoken_chunks(n_kv_head, position):
    torch.manual_seed(14)
    model = GPT(make_cfg(n_kv_head=n_kv_head, position=position)).eval()
    tokens = torch.randint(0, 37, (2, 13))
    reference = model(tokens).logits
    cache = model.make_cache(2)
    chunks = [(0, 5), (5, 8), (8, 9), (9, 13)]
    got = []
    with torch.inference_mode():
        for lo, hi in chunks:
            out = model(tokens[:, lo:hi], static_cache=cache)
            got.append(out.logits)
    assert cache.length == 13
    assert torch.allclose(reference, torch.cat(got, 1), atol=3e-5)
    cache.reset()
    got2 = model(tokens[:, :3], static_cache=cache).logits
    assert torch.allclose(got2, reference[:, :3], atol=3e-5)
    for key in cache.keys:
        assert key.shape[1] == n_kv_head  # stores groups, not repeated attention heads
    with pytest.raises(ValueError, match="longer|capacity"):
        model(torch.randint(0, 37, (2, 20)), static_cache=cache)


def test_fused_projection_identical_to_unfused_including_gradients():
    torch.manual_seed(30)
    a = GPT(make_cfg(fused_qkv=False)).train()
    b = GPT(make_cfg(fused_qkv=True)).train()
    from copy import deepcopy
    packed = deepcopy(a.state_dict())
    for index in range(a.config.n_layer):
        prefix = f"blocks.{index}.attn."
        qs = [prefix + field + ".weight" for field in ("q_proj", "k_proj", "v_proj")]
        packed[prefix + "qkv_proj.weight"] = torch.cat([packed.pop(k) for k in qs], 0)
    b.load_state_dict(packed)
    ids = torch.randint(0, 37, (2, 12))
    labels = torch.randint(0, 37, ids.shape)
    loss_a = a(ids, labels).loss
    loss_b = b(ids, labels).loss
    assert torch.allclose(loss_a, loss_b, atol=3e-6)
    loss_a.backward()
    loss_b.backward()
    for i in range(a.config.n_layer):
        attn_a = a.blocks[i].attn
        attn_b = b.blocks[i].attn
        refs = [attn_a.q_proj.weight.grad, attn_a.k_proj.weight.grad, attn_a.v_proj.weight.grad]
        assert torch.allclose(attn_b.qkv_proj.weight.grad, torch.cat(refs), atol=3e-6)


def test_greedy_static_dynamic_uncached_and_context_rollover():
    torch.manual_seed(4)
    model = GPT(make_cfg(block_size=7)).eval()
    prompt = torch.randint(0, 37, (2, 5))
    def run(**kwargs):
        return generate(model, prompt, max_new_tokens=12, temperature=0, **kwargs)
    static = run(use_cache=True, cache_mode="static")
    dynamic = run(use_cache=True, cache_mode="dynamic")
    uncached = run(use_cache=False)
    assert torch.equal(static, dynamic)
    assert torch.equal(static, uncached)


def test_last_logits_equal_full_logits():
    model = GPT(make_cfg()).eval()
    x = torch.randint(0, 37, (2, 9))
    out = model(x).logits[:, -1:]
    assert torch.allclose(out, model(x, logits_last_only=True).logits)
    with pytest.raises(ValueError):
        model(x, x, logits_last_only=True)


def test_vectorized_cpu_batching_and_device_storage_exact_rng():
    data = np.arange(130, dtype=np.uint32)
    rng1 = torch.Generator().manual_seed(123)
    rng2 = torch.Generator().manual_seed(123)
    a = TokenBatcher(data, block_size=11, batch_size=7, device=torch.device("cpu"), storage="cpu")
    b = TokenBatcher(data, block_size=11, batch_size=7, device=torch.device("cpu"), storage="device")
    for _ in range(3):
        x, y = a.next(rng1)
        xx, yy = b.next(rng2)
        assert torch.equal(x, xx)
        assert torch.equal(y, yy)
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_cache_validations():
    model = GPT(make_cfg()).eval()
    cache = model.make_cache(2)
    with pytest.raises(ValueError, match="batch"):
        model(torch.randint(0, 37, (1, 3)), static_cache=cache)
    with pytest.raises(ValueError):
        model(torch.randint(0, 37, (2, 3)), static_cache=cache,
              past_key_values=((torch.zeros(2, 2, 1, 8), torch.zeros(2, 2, 1, 8)),) * 2)


def test_activation_checkpointing_agrees_with_plain_backward():
    torch.manual_seed(123)
    a, b = GPT(make_cfg()).train(), GPT(make_cfg()).train()
    b.load_state_dict(a.state_dict())
    b.gradient_checkpointing = True
    tokens = torch.randint(0, 37, (2, 11))
    targets = torch.randint(0, 37, (2, 11))
    loss_a = a(tokens, targets).loss
    loss_b = b(tokens, targets).loss
    assert torch.allclose(loss_a, loss_b, atol=3e-6)
    loss_a.backward()
    loss_b.backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=5e-6), name


def test_fused_qkv_lora_trainable_and_zero_initialized():
    from llmfs.finetuning.lora import apply_lora, LoRALinear
    torch.manual_seed(5)
    model = GPT(make_cfg()).eval()
    x = torch.randint(0, 37, (2, 5))
    orig = model(x).logits.detach()
    names = apply_lora(model, rank=2, alpha=4)
    assert names == [f'blocks.{i}.attn.qkv_proj' for i in range(2)]
    assert isinstance(model.blocks[0].attn.qkv_proj, LoRALinear)
    assert torch.allclose(model(x).logits, orig, atol=1e-6)
    assert all('.A' in name or '.B' in name for name, p in model.named_parameters()
               if p.requires_grad)
