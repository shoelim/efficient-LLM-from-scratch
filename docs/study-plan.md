# LLM from Scratch — Study Plan and Repository Specification

**Mission:** Build a clean, fully tested, reproducible **decoder-only language model** from raw text to pretraining, evaluation, inference, and basic post-training. Everything important is derived, implemented, and verified step by step.

**Primary learning path:** Raschka's *Build a Large Language Model (From Scratch)* + ARENA's *Transformers from Scratch* + Stanford CS336 *Language Modeling from Scratch*. Karpathy's GPT-2 reproduction/nanochat, Neel Nanda's tutorials, and BuildAnLLM are comparison/reference materials, not code to copy into the project.

**Suggested pace:** 16 core weeks at 6–8 focused hours/week (adjust based on Python fluency and GPU access); 4–8 additional weeks for systems, scaling, richer data pipelines, and advanced post-training. Milestones, not dates, govern progress.

## 1. What “from scratch” means

- Write our own data preparation, byte-level BPE tokenizer, datasets/batching, embeddings, attention, Transformer blocks, language modeling loss, optimizer implementation, training driver, checkpointing, sampling, evaluations, and LoRA/SFT extensions.
- Use Python, NumPy when instructive, and **PyTorch tensors, autograd, and device backends** as numerical primitives. We do not attempt a custom CUDA runtime or autograd engine in v1; optional exercises explore gradient derivations and custom kernels later.
- Do not train using Hugging Face `Trainer`, pretrained model weights, `nn.Transformer`, or `nn.MultiheadAttention`. A pretrained GPT-2 checkpoint may be **loaded into our implementation for a verification experiment**, never used to claim scratch pretraining.
- Keep every educational implementation readable. Accelerated variants are **separate** and must demonstrate equivalence to the reference version.
- Implement first, inspect reference solutions second. Record licenses/provenance for data and source code.

## 2. Final deliverables

1. A `pip`/`uv`-installable Python package, CLI commands, configs, documentation, and continuous tests.
2. A character-level GPT-like model trained on a toy corpus (pipeline proof).
3. A byte-level BPE tokenizer implemented from first principles and tested on Unicode/edge cases.
4. A 5–10M-parameter subword decoder-only LM trained on a controlled corpus such as TinyStories; all weights randomly initialized.
5. Evaluation report with held-out loss, perplexity, sampling examples, throughput, ablations, baselines, and limitations.
6. Exact checkpoint resume (within the chosen deterministic environment), an implementation with and without KV caching, and profiling.
7. SFT and LoRA modules; a final release with experiment records and model card.
8. Optional stretch: GPT-2-small-compatible (~124M) architecture, modern architectural variants, more advanced GPU systems and data engineering, alignment/reasoning RL.

The goal is correctness and transparency first, **not an ungrounded claim of GPT-2-level generation quality** under a small compute budget.

## 3. Core learning loop: each lesson follows the same six steps

1. **Question:** What problem is this component solving?
2. **Mathematics:** Write definitions, objectives, tensor shapes, complexity, and assumptions.
3. **Minimal implementation:** Write it without looking at a reference solution.
4. **Verification:** Unit/property tests, hand calculations, finite-difference gradients or reference comparisons.
5. **Experiment:** Run a small reproducible exercise; record what happens and why.
6. **Cleanup:** Make a focused commit, update docs and tests, and link results.

A module is complete only when it has a working implementation, numerical tests, a brief derivation, a reproducible demonstration, and clear code comments documenting shapes and conventions.

## 4. 16-week core roadmap

### Week 0 — Python and repository foundations
- Refresh lists/dictionaries/tuples, iteration, comprehensions, slicing, functions, classes, `dataclass`, type hints, pathlib, iterators, exceptions, virtual environments, and debugging.
- Create `pyproject.toml`, `src/` package, `pytest`, Ruff, pinned dependencies/lockfile, `.gitignore`, and a minimal CI workflow.
- Practice: frequency counter, sliding context-target windows, iterating mini-batches, reproducible pseudo-random sampling.
- **Gate:** A clean install and `pytest` pass in an empty environment; tests for four pure Python helper functions.

### Week 1 — Probability and the language-modeling objective
- Derive the chain rule: `p(x_1,...,x_T) = product_t p(x_t | x_{<t})`.
- Explain autoregressive next-token prediction, maximum likelihood, teacher forcing, cross-entropy, empirical negative log likelihood, and perplexity.
- Implement count-based unigram, bigram, and simple smoothed bigram models; add a sampler and held-out evaluator.
- **Gate:** Hand-computed probabilities/loss match implementation; train and validation are distinct; uniform-logit cross-entropy is `log(V)`.

### Week 2 — PyTorch, tensor algebra, autograd
- Master Tensor shapes, broadcasting, `matmul`, `einsum`, `view` vs `reshape`, batch dimensions, devices, dtype, and basic numerical stability.
- Derive gradient of softmax-cross-entropy, verify with autograd and finite differences; inspect optimization of a tiny linear classifier.
- Use `torch.nn.Parameter` and `torch.nn.Module`; learn parameter registration and `state_dict`.
- **Gate:** Shape tests, gradient checks in double precision, manual one-step SGD update equals expected value.

### Week 3 — Text, document boundaries, and datasets
- Load raw UTF-8 documents; split by **document before** training/evaluation; keep a manifest and source/license details.
- Implement a character-level tokenizer, deterministic vocabulary, encode/decode, block/context windows, next-token target shift, padding/packing policy, and deterministic shuffling.
- Define treatment of BOS/EOS and document boundaries; make the effects observable.
- **Gate:** Roundtrip on supported text; `targets[t] == inputs[t+1]` in a contiguous training window; no leakage between data splits.

### Week 4 — Byte-level BPE tokenizer
- Implement the algorithm: start from byte symbols, count adjacent pairs, choose deterministic merges, apply ranked merges, encode/decode, handle special tokens and unseen Unicode.
- Test empty text, repeated text, multibyte characters, whitespace, bytes, and tie-breaking. Compare statistics with character tokenization.
- Serialize vocabulary/merges and verify a saved tokenizer reloads identically.
- **Gate:** Deterministic training, Unicode roundtrips, known toy pair-merging sequence, serialization parity, no dependence on a pretrained tokenizer.

### Week 5 — Attention from first principles
- Derive scaled dot-product attention `softmax(Q K^T / sqrt(d_k) + M) V`.
- Implement query/key/value projections, causal masking, explicit attention scores, numerically stable softmax, and single/multi-head attention.
- Check dimensions `(B,T,C) -> (B,H,T,d_head) -> (B,H,T,T) -> (B,T,C)`.
- **Gate:** A token's output is unchanged when only **future** tokens are modified; masked probabilities are zero; gradient checks pass; attention weights sum to one.

### Week 6 — GPT-2-style Transformer
- Implement token embeddings, learned positional embeddings, LayerNorm, residual connections, GELU feed-forward network, pre-norm Transformer blocks, final norm, and output projection.
- Build a minimal `GPTConfig` with explicit checks, initialization, dropout semantics, and optional tied output embeddings.
- Assemble `GPT.forward(tokens) -> logits` with logits shape `(B,T,V)`.
- **Gate:** Parameter count is explained, outputs have expected shapes, backpropagation works, model overfits a microscopic supervised sequence, serialization recovers identical outputs.

### Week 7 — Training algorithms and numerical correctness
- Implement cross-entropy via stable log-sum-exp; compare to PyTorch's `cross_entropy`.
- Implement AdamW explicitly (including bias correction and decoupled weight decay), then verify it against PyTorch on small deterministic tensors.
- Implement warm-up + cosine decay, gradient clipping, gradient accumulation, optimizer zeroing, and seed management.
- **Gate:** Tiny loss/optimizer tests; accumulated gradients agree with a mathematically equivalent full batch (dropout off); one-batch overfit test.

### Week 8 — First fully reproducible scratch training run
- Train a small character-level GPT on Tiny Shakespeare or another appropriately licensed small text corpus.
- Implement periodic held-out loss, token counts, tokens/sec, gradient norms, throughput, reproducible generation, and experiment logging.
- Save checkpoints with model, optimizer, scheduler, step, RNG states, config, and tokenizer/data identifiers.
- **Gate:** The model trains, validation loss is logged, checkpoint load works, uninterrupted and resumed runs agree under controlled deterministic conditions.

### Week 9 — Byte-level BPE model and a real pretraining corpus
- Prepare TinyStories (or a similar permissibly used data source); train a tokenizer on training text only and tokenize all splits with the frozen tokenizer.
- Suggested starter: vocab ~8k–10k, 4 layers, hidden size 256, 4 heads, context length 256, shared input/output embeddings, CPU smoke test first, GPU for training.
- Report dataset tokens, unique docs, tokenization stats, total parameters, training-token budget, effective batch size, compute and memory constraints.
- **Gate:** A real scratch subword LM completes training and passes independent validation; reproduce same configuration and seed.

### Week 10 — Generation and inference
- Implement greedy, temperature sampling, top-k, top-p (nucleus), EOS termination, context cropping, and deterministic seeded sampling.
- Explain mathematically what each sampling rule changes; test normalization and edge cases.
- Implement two generation loops: naïve full-context recomputation and incremental decoding with KV cache.
- **Gate:** Cached and uncached next-token logits agree within numerical tolerance in eval mode, with correct position indices and masks.

### Week 11 — Evaluation and diagnosis
- Add held-out NLL, perplexity, bits per byte (when comparing different tokenizers, with caveats), generation tests, and latency/memory/throughput metrics.
- Compare count-based models, a small Transformer, and optional published baselines on precisely defined data and tokenizer settings.
- Perform controlled ablations on context length, layers, heads, width, learning rate, and tokenization choices.
- **Gate:** A reproducible evaluation report; raw configs, seeds, token counts, learning curves, and limitations are included.

### Week 12 — GPT-2 compatibility and reference validation
- Compare our layer calculations with an independent implementation; write a weight-conversion test for an available pretrained GPT-2 checkpoint if practical.
- Be precise about GPT-2 bias terms, LayerNorm epsilon, GELU convention, positional embeddings, weight tying, and tensor layouts.
- Optional: load publicly available pretrained weights **only as a numerical compatibility test**; do not confuse with our pretrained model.
- **Gate:** Layerwise/logit parity to a reference within documented tolerance, or a written record of the exact unresolved discrepancy.

### Week 13 — Modernize one component at a time
- Add RMSNorm, RoPE, SwiGLU, and optionally grouped-query attention; compare against the GPT-2-style baseline.
- Discuss positional extrapolation and cache handling for RoPE; verify rotations and head/group indexing.
- Keep variants separate and configured explicitly; preserve the original reference path.
- **Gate:** Unit tests for every variant and **matched** experiment configurations; no unexplained replacement of multiple components at once.

### Week 14 — Efficiency without hiding the model
- Profile the baseline, compare manual attention with PyTorch's `scaled_dot_product_attention` in matching conditions.
- Add mixed precision, optional compilation, activation checkpointing, and throughput benchmarks as separate, switchable options.
- Measure speed **and** peak memory, and compare output/gradient differences under the appropriate numerical tolerances.
- **Gate:** Clearly documented benchmark environment and correctness-performance table.

### Week 15 — Post-training: SFT and LoRA
- Build prompt/response formatting, correct label masking, train/validation split, tokenization, supervised fine-tuning, and evaluation.
- Derive LoRA `W' = W + (alpha/r) B A`, freeze base parameters, train adapters, and test merge/unmerge numerical equivalence.
- Separate base LM evaluation from instruction-following evaluation; make no exaggerated claims about alignment.
- **Gate:** Working SFT training loop, verified target mask, gradient isolation, LoRA parameter counts, and before/after examples.

### Week 16 — Release and research readiness
- Run all tests in CI; document all commands, supported hardware, runtime/configuration, datasets/licenses, known bugs, and results.
- Prepare a model card, architecture explanation, derivations, reproducibility guide, experiment comparison, and clean tagged release (`v1.0`).
- **Gate:** A new contributor can install the repository, run tests, train a tiny model, generate output, and reproduce an evaluation without reading source code first.

## 5. Stretch curriculum (not required for v1.0)

1. **Systems:** custom GPU kernels/Triton FlashAttention, DDP/FSDP, memory accounting, communication and scaling (CS336 Assignment 2).
2. **Scaling and evaluation:** training FLOPs, data/parameter budgets, scaling fits, robust evaluations (CS336 Assignment 3).
3. **Data engineering:** deduplication, filtering, language ID, contamination checks, dataset mixing, provenance, PII handling (CS336 Assignment 4).
4. **Advanced architectures:** ALiBi, GQA/MQA, mixture-of-experts routing, auxiliary losses, more recent Transformer variants; do not label a loose variant an exact reproduction.
5. **Advanced post-training:** DPO/preference learning, reward modeling, or verifiable-reward RL, with explicit problem formulation (CS336 Assignment 5).
6. **Research extension:** compare autoregressive and flow/diffusion-based language modeling using *matched datasets, tokenization, model budgets, evaluation, and inference cost*.

## 6. Repository layout

The accompanying repository implements the core model and experimental infrastructure. The following paths are current (stretch assignments are **not** implied to be implemented):

```text
llm-from-scratch/
├── README.md
├── CONTRIBUTING.md
├── pyproject.toml
├── docs/{lessons.md,study-plan.md,efficiency.md}
├── configs/
├── src/llmfs/
│   ├── tokenization/
│   ├── data/
│   ├── model/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── finetuning/
│   └── utils/
├── scripts/
├── tests/
├── notes/
├── exercises/
├── experiments/
├── data/       # generated, ignored by Git
└── runs/       # generated, ignored by Git
```

The reference mathematical notes live in `notes/` and the executable command examples are in the root `README.md`. Additional modules listed in the stretch curriculum are future work.

## 7. Quality checklist for every module

- [ ] Definitions, objectives, assumptions, and tensor shapes documented.
- [ ] Unit tests for ordinary, edge, and failure cases.
- [ ] Numerical equivalence or analytical sanity checks where applicable.
- [ ] Determinism/reproducibility explicitly documented.
- [ ] Pure functions or small modules with clear interfaces; no hidden global state.
- [ ] Reference implementation kept distinct from optional optimized implementation.
- [ ] Training/evaluation data separated at the **document** level; tokenizer fitted on training only.
- [ ] Runtime and memory behavior understood; huge datasets/weights not checked into Git.
- [ ] One focused commit + short explanatory note.

## 8. Resource map

### Core, in the order we use them

- **Sebastian Raschka, _Build a Large Language Model (From Scratch)_**: https://www.sebastianraschka.com/llms-from-scratch/ — Chapters 2 (data), 3 (attention), 4 (GPT), 5 (pretraining), 6–7 (fine-tuning), plus PyTorch appendix; excellent primary sequential guide.
- **Stanford CS336 (Spring 2026), _Language Modeling from Scratch_**: https://cs336.stanford.edu/ — rigorous engineering/programming problem sets; Assignment 1 first, 2–5 later. Assignment 1 repo: https://github.com/stanford-cs336/assignment1-basics
- **ARENA, _Transformers from Scratch_ (Chapter 1.1)**: https://learn.arena.education/chapter1_transformer_interp/ — tensor-level intuition, reference compatibility, attention and transformer exercises.
- **Andrej Karpathy, _Let's reproduce GPT-2 (124M)_**: https://www.youtube.com/watch?v=l8pRSuU81PU — hands-on follow-up when our small model already works.
- **Andrej Karpathy, nanochat**: https://github.com/karpathy/nanochat — an *engineering reference* for the full model-building pipeline after learning and implementing the individual pieces.
- **BuildAnLLM**: https://github.com/jammastergirish/BuildAnLLM#resources — source of the resource collection, visualization and architecture ideas; do not copy its UI-first structure into v1.

### Background and later work

- **3Blue1Brown's LLM explanations** and Jay Alammar's **Illustrated Transformer**: conceptual background to be read before attention.
- **Neel Nanda's transformer walkthroughs** and the remaining ARENA chapters: interpretability once a functioning model exists.
- **Sebastian Raschka's Big LLM Architecture Comparison**: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison — reference for architectural extensions, not a recipe to add every feature immediately.
- Core papers: Vaswani et al. (Transformer); Radford et al. (GPT-2); Touvron et al. (LLaMA); modern architecture papers only when adding the corresponding feature.

## 9. First working session: a concrete checklist

- [ ] Start a new repository and write a two-paragraph `README.md` mission.
- [ ] Initialize a version-controlled Python environment (Python, PyTorch, pytest, Ruff, lockfile).
- [ ] Set up `src/llmfs/`, `tests/`, `notes/`, `exercises/`, and `scripts/`.
- [ ] Write `exercises/00_python_basics.py`: a function to count token frequencies and one to produce shifted `(context, target)` windows.
- [ ] Write `tests/test_python_basics.py`: empty input, repeated symbols, exact target offsets.
- [ ] Write `notes/01_autoregressive_lm.md`: the chain rule, one simple conditional probability table, and cross-entropy calculated by hand.
- [ ] Create and commit `week00-python-and-setup`; do not start the Transformer yet.

**Core philosophy:** derive → implement → test → experiment → document → commit. Progress is earned by passing a correctness gate, not by watching a given number of videos.
