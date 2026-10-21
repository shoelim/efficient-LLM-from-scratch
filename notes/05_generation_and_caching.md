# 05 — Autoregressive generation and KV caching

## Generating one token at a time

Given the current prefix `x_0:t`, compute next-token logits `z_t` and sample
`x_{t+1}`. Repeat, appending sampled tokens. This is different from teacher
forcing because the model now conditions on its own earlier outputs.

- **Greedy:** take `argmax_j z_j`; deterministic but can be repetitive.
- **Temperature τ > 0:** draw from `softmax(z/τ)`. As τ→0, mass tends toward
  the largest logit; τ=0 is implemented directly as greedy.
- **Top-k:** forbid all but k highest scores, renormalize the rest.
- **Top-p:** sort probabilities, retain the smallest prefix containing
  probability mass at least p, and renormalize.

We apply temperature, then top-k, then top-p. The sampler uses a user-provided
random generator so repeated experiments with the same seed are comparable.

## Why caching works

For causal self-attention, the keys and values of preceding tokens are
unchanged by new tokens. Recomputing them at every generation step is wasted
work. Each layer caches keys and values with shape
`(batch, n_kv_heads, cached_time, head_dim)`. On a new token, update these
caches and attend to the cached sequence. The new query uses the appropriate
absolute position for learned embeddings or RoPE.

**Context overflow:** a common incorrect optimization is to remove the oldest
KV pair and continue as if the remaining keys had been produced with shifted
absolute positional embeddings. For learned positions, this is wrong. This
implementation re-prefills on the last `block_size` tokens when the cache
fills, matching the reference uncached sliding-context computation. It is
correct but not the fastest possible streaming sampler.

## Checks

`tests/test_model.py` concatenates all single-token cached logits and checks
numerical parity with a full forward pass. `tests/test_sampling.py` asserts
*identical sampled sequences* across cache/no-cache, including overflow and
both learned positions and RoPE. If dropout is enabled, call `eval()` for
inference so the forward passes have deterministic semantics.

## Exercises

1. For logits `[0, log(2), log(4)]`, write the softmax and top-1 probabilities.
2. Derive top-p retention when probabilities are `[0.55, 0.3, 0.1, 0.05]`
   and p=0.8.
3. Remove absolute cache offsets and predict which parity test will fail.
4. Benchmark uncached vs cached generation for sequences shorter than the
   context window, then force an overflow and interpret the timing change.
