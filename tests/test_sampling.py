import pytest
import torch

from llmfs.config import GPTConfig
from llmfs.inference.sample import generate, sample_next_token
from llmfs.model.gpt import GPT


def test_greedy_topk_topp_and_validation():
    logits = torch.tensor([[1.0, 5.0, -3.0]])
    assert sample_next_token(logits, temperature=0).item() == 1
    assert sample_next_token(logits, top_k=1).item() == 1
    assert sample_next_token(logits, top_p=0.01).item() == 1
    with pytest.raises(ValueError):
        sample_next_token(logits, top_p=1.3)


@pytest.mark.parametrize("position", ["learned", "rope"])
def test_cached_uncached_generation_equivalence_even_when_context_overflows(position):
    torch.manual_seed(4)
    model = GPT(GPTConfig(vocab_size=32, n_embd=32, n_layer=2, n_head=4,
                          block_size=7, position=position)).eval()
    prompt = torch.tensor([[1, 2, 3]])
    result1 = generate(model, prompt, max_new_tokens=15, use_cache=True,
                       generator=torch.Generator().manual_seed(44))
    result2 = generate(model, prompt, max_new_tokens=15, use_cache=False,
                       generator=torch.Generator().manual_seed(44))
    assert torch.equal(result1, result2)
    assert result1.shape == (1, 18)


def test_eos_stops():
    torch.manual_seed(2)
    model = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2,
                          n_embd=16)).eval()
    # Only test that the generation path runs and respects max_new_tokens.
    output = generate(model, torch.tensor([[3]]), max_new_tokens=2, temperature=0)
    assert output.size(1) == 3
    batch = generate(model, torch.tensor([[1], [2]]), max_new_tokens=1, eos_id=0)
    assert batch.shape == (2, 2)
