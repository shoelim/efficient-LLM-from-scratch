import pytest
import torch

from llmfs.config import GPTConfig
from llmfs.finetuning.lora import (LoRALinear, adapter_state_dict, apply_lora,
                                   load_adapter_state_dict)
from llmfs.finetuning.sft import collate_sft, encode_example
from llmfs.model.gpt import GPT
from llmfs.tokenization.char import CharTokenizer


def test_lora_zero_initialization_gradient_isolation_merge_and_reload():
    torch.manual_seed(7)
    model = GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=16))
    x = torch.randint(0, 20, (1, 8))
    baseline = model(x).logits.detach().clone()
    names = apply_lora(model, rank=2, alpha=4)
    assert len(names) == 4
    assert all(p.requires_grad == (".A" in name or ".B" in name)
               for name, p in model.named_parameters())
    assert torch.allclose(model(x).logits, baseline, atol=1e-6)
    adapter = model.blocks[0].attn.q_proj
    assert isinstance(adapter, LoRALinear)
    with torch.no_grad():
        adapter.B.normal_(std=0.04)
    unmerged = model(x).logits.detach().clone()
    saved = adapter_state_dict(model)
    adapter.merge()
    assert torch.allclose(model(x).logits, unmerged, atol=3e-6)
    adapter.unmerge()
    assert torch.allclose(model(x).logits, unmerged, atol=3e-6)
    with torch.no_grad():
        adapter.B.zero_()
    load_adapter_state_dict(model, saved)
    assert torch.allclose(model(x).logits, unmerged, atol=1e-6)


def test_sft_masks_prompt_but_not_first_response():
    tok = CharTokenizer.train(["abcxyz ?."])
    x, y = encode_example("ab", "xyz", tok, 12)
    assert len(x) == len(y)
    assert y[0] == -100
    assert y[1] == tok.encode("x")[0]
    assert y[-1] == tok.eos_id
    xbatch, ybatch = collate_sft([(x, y), ([tok.eos_id], [tok.encode("a")[0]])],
                                 eos_id=tok.eos_id, device=torch.device("cpu"))
    assert (ybatch[1, 1:] == -100).all()
    with pytest.raises(ValueError, match="too long"):
        encode_example("abc", "xyz", tok, 2)
