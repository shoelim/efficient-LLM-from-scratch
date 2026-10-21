# 02 — Attention tensor algebra, masks, and gradients

A training batch of `B` sequences, each containing `T` tokens, is represented
by hidden states `X∈R^{B×T×C}`. For `H` heads of dimension `D=C/H`:

```
X             (B, T, C)
Q = X W_Q     (B, T, H*D) -> (B, H, T, D)
K = X W_K     (B, T, K*D) -> (B, K, T, D)  [K=H ordinarily]
V = X W_V     (B, T, K*D) -> (B, K, T, D)
Q K^T         (B, H, T, T)
softmax @ V   (B, H, T, D)
merge heads   (B, T, C)
output W_O    (B, T, C)
```

For position i attending to position j, the (scaled) logit is

\[
s_{ij}=\frac{q_i^\top k_j}{\sqrt{D}},\qquad
\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{\ell\le i}e^{s_{i\ell}}},\qquad
y_i=\sum_{j\le i}\alpha_{ij}v_j.
\]

We constrain `j <= i` to prevent the next-token model from reading future
labels. In code, a boolean matrix `allowed` permits a key if its absolute
position is no greater than the query's position. Masking is **before** the
softmax, with negative infinity (or the lowest finite dtype number) for
invalid keys. Softmax operates on the last axis, independently for each query.

## Why divide by sqrt(D)?

If each component of q and k has independent zero-mean unit-variance entries,
then `Var(q·k) ≈ D`; dividing by `sqrt(D)` keeps the score scale roughly
constant as head width increases. This is an approximate motivation, not an
assumption true for all trained attention projections.

## What is cached?

At autoregressive step t, old keys and values can be reused because they do
not depend on future input. A cache holds `K_0:t, V_0:t`, separately for each
layer. A new token supplies only `Q_t, K_t, V_t`. The mask must know that the
new query has **absolute position t**, not relative position zero. Incorrect
cache offsets often pass tests on the first token and fail on token two.

**Grouping:** GQA uses `K` query-independent key/value heads and `H` query
heads with `K` dividing `H`. The implementation repeats each KV head over
`H/K` query heads *for computing attention*, but stores the compact KV tensors
in the cache. The model still maps back to `C` dimensions.

## The numerical checks are not optional

- `tests/test_model.py::test_causality_future_token_change_cannot_affect_past`
  modifies only later tokens and checks that earlier logits are unchanged.
- `test_manual_sdpa_parity_forward_and_gradients` compares our matrix-based
  implementation to PyTorch's scaled dot product attention, with both forward
  and gradient tolerances.
- `test_cache_logit_equivalence` checks that concatenated logits from
  one-token incremental evaluation agree with a full-sequence evaluation.
- Standard attention requires `O(BHT²D)` dot products and a `T×T` score
  matrix. The explicit implementation stores attention probabilities. SDPA can
  use optimized kernels that avoid materializing this matrix on some devices.

## Exercises

1. Work through a two-token, one-head example with q=k=v scalars.
2. Write a 3×3 boolean causal matrix and manually zero the forbidden scores.
3. Derive `QK^T` shape using batch matrix multiplication.
4. Remove the cache offset from `attention.py`, run the cached parity test,
   observe the failure, then restore the correct code.
