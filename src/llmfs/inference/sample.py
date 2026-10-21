"""Autoregressive sampling, top-k/p filtering, and optional incremental KV caching."""

import torch
import torch.nn.functional as F

from llmfs.model.gpt import GPT


def sample_next_token(logits: torch.Tensor, *, temperature: float = 1.0,
                      top_k: int | None = None, top_p: float | None = None,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """Accept (batch,vocab) logits; return (batch,1) IDs. Temperature=0 is greedy."""
    if temperature < 0 or top_k is not None and top_k < 1:
        raise ValueError("temperature must be >= 0 and top_k >= 1")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if logits.ndim != 2:
        raise ValueError("Expected (batch,vocab) logits")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits.float() / temperature
    if top_k is not None:
        values, _ = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1)
        scores = scores.masked_fill(scores < values[:, -1:], -torch.inf)
    if top_p is not None and top_p < 1:
        sorted_logits, indices = torch.sort(scores, descending=True, dim=-1)
        probabilities = F.softmax(sorted_logits, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        # Keep the smallest prefix whose probability mass reaches/exceeds top_p.
        excluded = cumulative - probabilities >= top_p
        sorted_logits = sorted_logits.masked_fill(excluded, -torch.inf)
        scores = torch.empty_like(scores).scatter_(dim=-1, index=indices, src=sorted_logits)
    probabilities = F.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.inference_mode()
def generate(model: GPT, prompt: torch.Tensor, *, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None, eos_id: int | None = None,
             use_cache: bool = True, cache_mode: str = "static",
             generator: torch.Generator | None = None,
             precision: str = "fp32") -> torch.Tensor:
    """Incrementally decode with preallocated KV storage (default).

    Batches must have equally long *unpadded* prompts; no padding masks are
    supported. With EOS stopping, finished rows emit EOS until all rows finish.
    When the context window fills, re-prefill the most recent window. This is
    necessary for exact sliding-window equivalence with learned positions and
    avoids incorrectly reusing rotated keys from earlier context positions.
    """
    if prompt.ndim != 2 or prompt.shape[1] < 1 or max_new_tokens < 0:
        raise ValueError("Prompt must be nonempty (B,T); max_new_tokens >= 0")
    if cache_mode not in ("static", "dynamic"):
        raise ValueError("cache_mode must be static or dynamic")
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    device = prompt.device
    if precision != "fp32" and device.type != "cuda":
        raise ValueError("Low-precision generation requires CUDA")
    if max_new_tokens == 0:
        return prompt.clone()
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[precision]
    static = model.make_cache(prompt.size(0), dtype=dtype) if use_cache and cache_mode == "static" else None
    legacy_cache = None
    pieces = [prompt]
    done = torch.zeros(prompt.size(0), device=device, dtype=torch.bool)
    was_training = model.training
    model.eval()
    try:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "fp32"):
            for _ in range(max_new_tokens):
                cached_len = static.length if static is not None else (
                    legacy_cache[0][0].size(-2) if legacy_cache is not None else 0)
                if not use_cache or cached_len == 0 or cached_len >= model.config.block_size:
                    # Only concatenate history when a context re-prefill is actually required.
                    context = torch.cat(pieces, dim=1)[:, -model.config.block_size:]
                    if static is not None:
                        static.reset()
                        out = model(context, static_cache=static, logits_last_only=True)
                    else:
                        out = model(context, use_cache=use_cache, logits_last_only=True)
                        legacy_cache = out.past_key_values
                else:
                    last = pieces[-1]
                    if static is not None:
                        out = model(last, static_cache=static, logits_last_only=True)
                    else:
                        out = model(last, past_key_values=legacy_cache,
                                    use_cache=True, logits_last_only=True)
                        legacy_cache = out.past_key_values
                token = sample_next_token(out.logits[:, -1, :], temperature=temperature,
                                          top_k=top_k, top_p=top_p, generator=generator)
                if eos_id is not None:
                    token = torch.where(done[:, None], torch.full_like(token, eos_id), token)
                    done |= token[:, 0].eq(eos_id)
                pieces.append(token)
                if eos_id is not None and bool(done.all()):
                    break
    finally:
        model.train(was_training)
    return torch.cat(pieces, dim=1)
