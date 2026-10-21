# Fast-path design and mathematical invariants

The objective remains next-token maximum likelihood:

\[
\mathcal L(\theta)=-\frac1{BT}\sum_{b=1}^B\sum_{t=1}^T
\log p_\theta(x_{b,t+1}\mid x_{b,\le t}).
\]

The training code implements standard causal attention; optimization must not
change this objective or leak future token information. Separate Q,K,V and
manual masked softmax are retained for **parity tests**, not as the fast
training default.

## SDPA and masking: subtle but critical

For queries Q of length T and K,V of length S,

\[
A(Q,K,V)=\mathrm{softmax}\bigl(QK^\top/\sqrt{d_h}+M\bigr)V.
\]

- **Training/prefill (no past)**: `scaled_dot_product_attention(...,
  is_causal=True)`. The library selects FlashAttention or another supported
  backend. No explicit `[T,T]` Boolean mask is materialized in our Python code.
- **One-token cached decode (past length P)**: query length 1, key length P+1.
  Every key is allowed; use `is_causal=False`, not `True` with a non-square
  Q/K layout. That common mistake can permit attention only to the first key.
- **Several tokens appended to an existing cache**: query positions P...P+T-1
  must attend to key positions up to the absolute query position. Construct an
  offset-aware Boolean mask `[T,P+T]` for this less common case.
- **Grouped-query attention**: native SDPA's `enable_gqa=True` is used on CUDA,
  otherwise KV heads are expanded for the actual SDPA operation. The
  **persistent** inference cache always stores just `n_kv_head` groups.

Attention dropout is **always zero in `eval()`**, regardless of configuration.
Automated tests compare full and cached logits for learned positions/RoPE,
various K/V group counts and differently sized cached chunks.

## Static inference cache: memory and correctness

For batch B, layers L, context C, query heads Hq, KV heads Hkv, and
head dimension d, a static cache stores

\[
2 B L H_{kv} C d \;\text{elements}
\]

rather than 2 B L Hq C d. In bf16 this uses about
`4 * B * L * Hkv * C * d` bytes; memory is allocated once for the chosen
maximum context. A *growing* cache repeatedly copies previously stored keys
and values on concatenation, whereas the static version writes only the newly
produced tensors. Nevertheless, read/prefill cost, SDPA dispatch overhead and
cache allocation can make the static version slower on very small CPU tasks.

The implementation does not shift cached RoPE keys or learned absolute
positions when a window overflows: instead it rebuilds a fresh cache from the
last C tokens. This guarantees equivalence with ordinary full-context
recomputation, at the cost of intermittent O(C²) prefill. A production streaming
model with RoPE, ring cache or sliding-window attention needs different
position semantics and explicit correctness tests.

## Training fast paths

- Fused QKV combines three linear projections into one, yielding exactly the
  same outputs with packed weights; verified against separately projected
  Q,K,V in forward and backward tests.
- `torch.optim.AdamW(fused=True)` is selected by `optimizer: auto` on CUDA;
  native `torch.optim.AdamW` is selected on CPU/MPS. The independent
  `ScratchAdamW` implementation is retained for understanding and testing.
- AMP autocasting keeps model parameters in float32. bf16 does not generally
  require gradient scaling; fp16 uses `torch.amp.GradScaler`.
- Gradient accumulation allows effective batch B × accumulation steps, and
  gradient clipping happens once after the accumulated, unscaled gradients.
- `torch.compile` and non-reentrant activation checkpointing are optional
  runtime controls; recompilation, compile latency, and memory/compute
  tradeoffs must be measured per workload.
- Training samples are vectorized shifted windows of a memory-mapped uint32
  corpus. CPU mode transfers only requested batches; GPU mode optionally
  stages the whole token array in accelerator int32 memory.

The trainer can be launched with standard PyTorch **DDP** using `torchrun`.
Each process samples independent token windows, gradients are averaged over
ranks by DDP, and gradient accumulation skips unnecessary all-reduces via
`no_sync()` on the intermediate microsteps. The checkpoint includes per-rank
RNG states, and a two-rank CPU resume-parity test is provided. Do **not**
combine DDP with `torch.compile` in this verified code path: the trainer
explicitly rejects that combination. CUDA/NCCL performance is unverified.

FSDP, tensor parallelism, sharded optimizer state, advanced asynchronous
dataloading and elastic restart remain future work.

## Evaluation and performance methodology

- Use *matched* tokenizer, data, model dimensions, sequence length, batch,
  temperature, context resets, precision and hardware.
- Always verify output parity (strict for greedy token decisions, numeric
  tolerances for logits), then report prefill latency, decode tokens/sec,
  training tokens/sec, activation memory, peak allocated GPU memory and run-to-run
  variation. GPU benchmarking requires synchronization and warmup.
- For `torch.compile`, report separate cold-start/compile and steady-state
  measurements. It is not measured by the included benchmark.
- A small CPU benchmark is only a mechanics sanity check, not a substitute for
  actual matched GPU benchmarking or statistical repetitions.
- Autoregressive training NLL or token perplexity must not be used to compare
  differently-tokenized models without careful normalization.

## Limits for scaling the repository

The instructional tokenization trainer repeatedly recounts all pairs. Input
parsing also materializes raw documents to perform reproducible document-level
splits. The `.npy` arrays themselves are memory-mapped during training. For
very large corpora, implement a scalable streaming/tokenizer preprocessor and
store token IDs in shards, with validation split at the document level. This
repository intentionally does not pretend that the reference tokenizer trainer
can prepare billions of tokens efficiently.

## External implementation references

- PyTorch documentation: [scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html),
  [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile),
  [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html),
  [AMP](https://docs.pytorch.org/docs/stable/amp.html).
- [Stanford CS336](https://cs336.stanford.edu/),
  [nanochat](https://github.com/karpathy/nanochat).
