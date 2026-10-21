# LLM from Scratch

**A readable decoder-only Transformer implementation with efficient PyTorch training and KV-cached inference.**

Build a language model end to end: prepare text, train a tokenizer, train a GPT-style model from random initialization, evaluate held-out loss, generate tokens, and fine-tune with supervised examples or LoRA. The reference implementations and numerical parity tests make it possible to study each optimization rather than treating it as a black box.

> This is a research and learning codebase, **not** a pretrained model or a production inference server. PyTorch provides tensors, autograd, and optimized attention kernels; the model, trainer, tokenization, and generation logic are implemented here.

## Features

| Area | Included |
|---|---|
| **Model** | Pre-norm decoder-only Transformer; learned positions or RoPE; LayerNorm/RMSNorm; GELU/SwiGLU; MHA/GQA; tied embeddings |
| **Reference & fast paths** | Manual or SDPA causal attention, separate or fused QKV, reference and heap-based byte-BPE encoding |
| **Training** | Scratch/PyTorch/fused AdamW, cosine learning-rate schedule, gradient accumulation, AMP, optional activation checkpointing and `torch.compile`, DDP via `torchrun` |
| **Inference** | Prefill and token-by-token decoding, growing or preallocated KV cache, top-k/top-p/temperature sampling, inference-only `safetensors` export |
| **Adaptation** | Response-only supervised fine-tuning and LoRA |
| **Verification** | Unit and integration tests, cache/reference parity tests, CPU smoke scripts, benchmarking utilities |

## Install

Use Python 3.10+ and PyTorch 2.5+. For NVIDIA GPUs, install a compatible CUDA-enabled PyTorch build using the [official instructions](https://pytorch.org/get-started/locally/) before installing this package.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

The installed CLI is `llmfs` (or `python -m llmfs`). All commands below work with the included miniature text example on CPU; no model weights are bundled.

## Quick start: text to generated tokens

```bash
# 1. Prepare documents, train a character vocabulary and save token arrays.
llmfs prepare --input examples/toy_corpus.txt --output data/demo --tokenizer char

# 2. Train a tiny Transformer from random weights (CPU smoke configuration).
llmfs train --data data/demo --config configs/smoke.yaml --output runs/demo

# 3. Measure held-out next-token loss and perplexity.
llmfs evaluate --checkpoint runs/demo/last.pt --data data/demo \
  --batch-size 4 --batches 5

# 4. Export weights, model configuration and tokenizer for inference.
llmfs export --checkpoint runs/demo/last.pt --output runs/demo-infer

# 5. Sample from the trained model using a static KV cache.
llmfs generate --checkpoint runs/demo-infer --prompt 'once upon' \
  --max-new-tokens 40 --temperature 0.8 --top-k 20
```

This example proves the pipeline runs; **a few updates on a toy corpus will not produce meaningful text**. Use fresh output directories for new runs. Training refuses to overwrite a populated run directory; to continue an interrupted run, provide `--resume runs/demo/last.pt` with the same data, config, and world size.

Run the complete CPU training/export/generation/benchmark smoke test in an isolated temporary directory:

```bash
bash scripts/smoke_efficient.sh
```

## Train on a larger corpus

```bash
# Treat blank-line-separated paragraphs or individual .txt files as documents.
llmfs prepare --input path/to/text/ --output data/corpus \
  --tokenizer bpe --vocab-size 512

# Single-device training; select an appropriate config for your hardware.
llmfs train --data data/corpus --config configs/efficient_cuda.yaml \
  --output runs/my-experiment

# Two GPUs, one DDP process per GPU:
torchrun --standalone --nproc_per_node=2 -m llmfs train \
  --data data/corpus --config configs/efficient_cuda.yaml \
  --output runs/my-ddp-experiment
```

`configs/efficient_cpu.yaml` is a CPU alternative. The CUDA config enables mixed-precision execution; `torch.compile` is opt-in. The included BPE **trainer** is a transparent learning implementation and is too slow for training a large vocabulary on multi-GB corpora. For large-scale use, prepare tokens with a scalable tokenizer/data pipeline and integrate the resulting IDs/metadata.

Resume training by repeating the original command with `--resume <run>/last.pt`; a run's training schedule, data fingerprint, and distributed world size must match. Only load PyTorch `.pt` training checkpoints from trusted sources. Inference exports use `safetensors` instead.

## Generate, fine-tune, and benchmark

```bash
# Compare decoding implementations on identical weights.
llmfs generate --checkpoint runs/demo-infer --prompt 'once upon' \
  --max-new-tokens 30 --temperature 0 --cache-mode dynamic
llmfs generate --checkpoint runs/demo-infer --prompt 'once upon' \
  --max-new-tokens 30 --temperature 0 --no-cache

# Response-only SFT with a LoRA adapter; this uses a toy instructional dataset.
llmfs sft --checkpoint runs/demo/last.pt --data examples/sft_toy.jsonl \
  --output runs/demo-lora --lora --rank 4 --alpha 8 --steps 20

# Report training and generation throughput on the current machine.
python scripts/benchmark_llm.py --device cpu --threads 1 --repeats 10
# Run on CUDA hardware, for example:
# python scripts/benchmark_llm.py --device cuda --precision bf16 \
#   --batch 4 --seq 256 --new-tokens 96 --output benchmarks/my_gpu.json
```

The preallocated cache avoids repeatedly concatenating K/V tensors, but it currently **re-prefills the window when context is full** to preserve exact truncated-window semantics. This can be expensive during long generation. GPU throughput, fused-kernel selection, and multi-GPU scaling depend on hardware and should be benchmarked rather than assumed.

## Repository layout

```text
src/llmfs/
  tokenization/    character and byte-level BPE tokenizers
  data/            document splitting, token arrays and batched windows
  model/           embeddings, attention, blocks and full GPT model
  training/        optimizer, scheduler, AMP/DDP trainer and checkpoints
  inference/       generation, KV cache and safetensors export
  finetuning/      supervised fine-tuning and LoRA
  evaluation/      losses, perplexity and count-based baselines
configs/           small CPU/GPU and architecture configurations
scripts/           smoke and benchmark commands
tests/             correctness and integration tests
notes/             derivations for the key components
exercises/         coding exercises with partial solutions
docs/              study plan, guided lessons and efficiency notes
examples/          tiny text and prompt/response samples
```

**Study it step by step:** [guided lessons](docs/lessons.md) · [16-week plan](docs/study-plan.md) · [math notes](notes/) · [optimization details](docs/efficiency.md).

**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for tests, coding conventions and reproducibility guidelines.

## Scope and limitations

The repository does **not** provide pretrained GPT-2 weight compatibility, verified large-scale model quality, FSDP, distributed inference, dynamic request batching, paged attention, quantization, speculative decoding or an HTTP serving stack. The bundled CPU examples and tests validate functional behavior, not competitive model performance; CUDA/NCCL paths have not been benchmarked in the development environment.

Influenced by [BuildAnLLM](https://github.com/jammastergirish/BuildAnLLM#resources), [Raschka's *Build a Large Language Model (From Scratch)*](https://www.sebastianraschka.com/llms-from-scratch/), [ARENA](https://learn.arena.education/chapter1_transformer_interp/), [Stanford CS336](https://cs336.stanford.edu/), and [nanochat](https://github.com/karpathy/nanochat). These are reading references, not runtime dependencies or copied source trees.

Released under the [MIT License](LICENSE). Bring your own suitably licensed training data; no external training corpus is bundled.
