# Course index: 17 checkpoints from empty Python to an LLM

**Method:** work through the assigned source and math notes, implement an
exercise without peeking at the solution, run the corresponding tests, and
record one result. The production implementation is fully provided; the
`exercises/` folder alone contains purposeful TODOs for deliberate practice.

| Week | Essential question | File to read / modify | Correctness gate |
|---|---|---|---|
| 0 | How do I write reliable Python? | `exercises/00_python_basics.py`; `tests/test_data.py` | Dict counting, windows, pytest green |
| 1 | What distribution does next-token MLE fit? | `notes/01_autoregressive_lm.md`; `evaluation/baselines.py` | Compute a bigram NLL by hand |
| 2 | How does backpropagation see tensor shapes? | `notes/02_attention_and_shapes.md`; `tests/test_model.py` | Autograd/finite difference agrees |
| 3 | Why split *documents* before training vocabulary? | `data/prepare.py`, `data/batches.py` | No document overlaps splits |
| 4 | What exactly is byte-level BPE? | `notes/03_tokenization.md`; `tokenization/bpe.py` | UTF-8 and serialization roundtrip |
| 5 | Why does attention require √d and causal masking? | `model/attention.py`, `notes/02_attention_and_shapes.md` | Future change cannot affect past |
| 6 | How is a GPT block constructed? | `model/gpt.py`, `model/block.py`, `model/layers.py` | Forward/backward/weights tie |
| 7 | How are optimization and LR schedules implemented? | `training/optim.py`, `notes/04_training_mathematics.md` | Match PyTorch AdamW |
| 8 | Can a small model actually train? | `training/train.py`, `scripts/smoke_efficient.sh` | Complete CLI end-to-end |
| 9 | What changes when moving from characters to subwords? | `configs/tinystories_bpe.yaml`, `data/prepare.py` | BPE corpus and manifests reproducible |
| 10 | How do sampling and KV caching work? | `inference/sample.py`; `notes/05_generation_and_caching.md` | Cached vs uncached parity |
| 11 | What does loss / perplexity mean on held-out text? | `evaluation/metrics.py`, `notes/07_reproducibility.md` | Evaluate independent held-out windows |
| 12 | Which differences prevent exact GPT-2 parity? | `configs/gpt2_small_architecture.yaml` | Produce a checklist, not a false parity claim |
| 13 | What do RoPE, GQA, RMSNorm and SwiGLU change? | `model/layers.py`, `model/attention.py` | Variant-specific tests |
| 14 | Can acceleration preserve correctness? | `scripts/benchmark_attention.py` | Compare manual vs SDPA timing and outputs |
| 15 | How is the response-only SFT objective formed? | `finetuning/sft.py`, `finetuning/lora.py` | Verify label mask, gradient isolation |
| 16 | Can someone reproduce a real result? | `experiments/README.md`, `tests/test_integration.py` | Results documented, not guessed |

## Suggested study session

1. **Before code:** write the equation and all input/output tensor shapes.
2. **Smallest possible example:** calculate a two-token or two-head case by hand.
3. **Implement independently:** use `exercises/` when one exists or clone a
   production function into a scratch file.
4. **Adversarial tests:** try empty input, mask boundaries, Unicode, long
   context, NaN gradients, reload, and a deliberately broken version.
5. **Experiment:** make exactly one controlled change; record configuration,
   held-out metric, wall-clock, and outcome.
6. **Git:** commit just the verified change with a short derivation or note.

Reading is not a substitute for implementing or evaluating. The accompanying
[study plan](study-plan.md) gives more reading assignments and rationale.
