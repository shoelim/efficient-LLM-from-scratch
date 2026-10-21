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

## Train on a real dataset: TinyStories (recommended)

**[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** is the recommended first dataset for training a small language model from random initialization. It contains short English stories with a relatively limited vocabulary, so it is easier to run meaningful language-model experiments before moving to broad web-text datasets. The raw data and pretrained weights are **not** included in this repository.

This walkthrough uses **20,000 training stories and 1,000 validation stories**. Both come from the dataset's existing, separate train and validation splits; this is a bounded starter experiment, **not** the full TinyStories benchmark.

### 1. Install the dataset and tokenizer dependencies

From the repository root, after creating/activating your virtual environment:

```bash
python -m pip install -e '.[dev,data]'
```

The `data` extra installs Hugging Face `datasets` (for downloading) and the Rust-backed `tokenizers` package (for practical BPE training). It **does not** download a pretrained tokenizer: we will train our own on the downloaded training stories.

### 2. Download the dataset

```bash
python scripts/download_tinystories.py \
  --output data/tinystories/raw \
  --train-stories 20000 \
  --validation-stories 1000 \
  --seed 1337
```

The downloader uses streaming and buffered shuffling, rejects exact duplicate stories within/across splits, and writes:

```text
data/tinystories/raw/
├── train.jsonl        # 20,000 training stories, one {"text": "..."} per line
├── validation.jsonl   # 1,000 separate validation stories
└── source.json        # provenance, row counts, seed, file checksums
```

A story is one JSON object per line, for example:

```json
{"text": "Once upon a time, there was a little bird."}
```

The downloader refuses to overwrite a nonempty output directory. To start a different experiment, use a new output path. For more reproducible datasets, optionally pass `--revision <HUGGING_FACE_COMMIT_SHA>` to pin the dataset source; the default follows the provider's current revision. Note that streamed buffered shuffling is **not** a uniform random sample from the entire corpus.

### 3. Train a tokenizer and prepare token arrays

```bash
llmfs prepare \
  --input data/tinystories/raw/train.jsonl \
  --validation-input data/tinystories/raw/validation.jsonl \
  --output data/tinystories/packed \
  --tokenizer fast-bpe \
  --vocab-size 2048
```

This step:

1. Trains a **2048-token byte-level BPE tokenizer from scratch** on training stories only.
2. Encodes both splits with that same frozen tokenizer, adding an end-of-document token after each story.
3. Saves the sequences and metadata in:

```text
data/tinystories/packed/
├── tokenizer.json     # learned vocabulary and merges
├── train.npy          # training token IDs (uint32)
├── val.npy            # held-out token IDs (uint32)
└── manifest.json      # sizes, source paths and integrity checksums
```

`fast-bpe` is the practical option for this experiment. For understanding BPE itself, `--tokenizer bpe` uses our pure-Python implementation but is considerably slower. The educational `--tokenizer char` option is also available. `llmfs prepare` holds documents in memory, so **do not use it to ingest the entire multi-million-story corpus** without redesigning the preprocessing pipeline for streaming/sharding.

### 4. Train the language model

**NVIDIA GPU with BF16 support (recommended for the starter experiment):**

```bash
llmfs train \
  --data data/tinystories/packed \
  --config configs/tinystories_starter_cuda.yaml \
  --output runs/tinystories-starter
```

This configuration uses a small four-layer Transformer with RoPE, RMSNorm, SwiGLU, and efficient scaled dot-product attention, initialized **from scratch**. It specifies 3,000 optimizer steps, which is a starter setting rather than a quality-optimal recipe. You can adjust batch size, context length, learning rate, and steps in the YAML configuration to fit your hardware and experiment.

**CPU alternative (primarily to verify the pipeline, not a quality benchmark):**

```bash
llmfs train \
  --data data/tinystories/packed \
  --config configs/tinystories_starter_cpu.yaml \
  --output runs/tinystories-cpu
```

The CPU config uses a smaller two-layer model and fewer steps. If your GPU lacks BF16 support, change `precision` and related settings in a copy of the CUDA YAML according to your hardware before training.

The training run writes checkpoints and logs in its output directory, including `last.pt`. The project deliberately prevents a new run from silently reusing a populated run directory. **Use a new directory for a new run** or resume an interrupted run (see below).

### 5. Evaluate on held-out stories

```bash
llmfs evaluate \
  --checkpoint runs/tinystories-starter/last.pt \
  --data data/tinystories/packed \
  --batches 40 \
  --batch-size 8
```

This reports held-out next-token negative log-likelihood and perplexity, estimated from sampled validation windows. Lower is generally better **when the tokenizer and evaluation procedure are held fixed**. The default validation evaluator is not an exhaustive full-corpus metric.

For a simple count-based reference on the same token sequence:

```bash
llmfs baseline --data data/tinystories/packed
```

### 6. Export and generate text

Export a smaller **inference-only** checkpoint (weights in `safetensors`, plus model configuration and tokenizer):

```bash
llmfs export \
  --checkpoint runs/tinystories-starter/last.pt \
  --output runs/tinystories-export
```

Generate a story using the optimized preallocated KV cache:

```bash
llmfs generate \
  --checkpoint runs/tinystories-export \
  --prompt 'Once upon a time, there was a little' \
  --max-new-tokens 150 \
  --temperature 0.8 \
  --top-k 40
```

For deterministic greedy decoding, set `--temperature 0`. You may also try `--top-p 0.9`, `--no-cache`, or `--cache-mode dynamic` to compare sampling and inference paths. Small models or very short runs may still produce poor text; don't infer model quality from the smoke test alone.

### 7. Resume training (optional)

To continue the **same run** after an interruption, use the same data, configuration, and output directory, supplying its checkpoint:

```bash
llmfs train \
  --data data/tinystories/packed \
  --config configs/tinystories_starter_cuda.yaml \
  --output runs/tinystories-starter \
  --resume runs/tinystories-starter/last.pt
```

Resume checks the dataset fingerprint and saved configuration. Do not use `--resume` to start a fundamentally different experiment; create a new directory instead. Load `.pt` checkpoints only from trusted sources.

### Try a smaller sample or your own text

To test the **download → preparation → training** path with less data, download a separate small subset (for example, `--train-stories 1000 --validation-stories 100 --output data/tinystories/small-raw`), prepare it into a different `small-packed` directory, and use the CPU config with a **fresh** run directory. This mainly checks that everything works; it won't yield good stories.

To use local text instead, provide either a `.jsonl` file with one `{"text": "..."}` document per line, a `.txt` file whose blank-line-separated paragraphs are documents, or a directory of `.txt` files:

```bash
llmfs prepare \
  --input path/to/my_corpus.txt \
  --output data/my-corpus \
  --tokenizer char

llmfs train \
  --data data/my-corpus \
  --config configs/smoke.yaml \
  --output runs/my-corpus-test
```

If you do **not** specify `--validation-input`, `prepare` makes a seeded document-level train/validation split before tokenizer training. If you already have a held-out dataset, supply it with `--validation-input` instead. Never use validation or test data to train the tokenizer.

**Dataset and licensing notes:** The downloaded data live under `data/` and are excluded from Git; training artifacts under `runs/` are also ignored. TinyStories lists the **CDLA-Sharing-1.0** license; review its dataset card and the license before distributing data.


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

`configs/efficient_cpu.yaml` is a CPU alternative. The CUDA config enables mixed-precision execution; `torch.compile` is opt-in. The included pure-Python BPE **trainer** is a transparent learning implementation and too slow for large datasets. For practical bounded subsets, install `.[data]` and use `--tokenizer fast-bpe`. For large-scale use, prepare tokens with a scalable tokenizer/data pipeline and integrate the resulting IDs/metadata.

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
examples/          tiny original offline text and prompt/response samples
```

**Study it step by step:** [guided lessons](docs/lessons.md) · [16-week plan](docs/study-plan.md) · [math notes](notes/) · [optimization details](docs/efficiency.md).

**Contributing:** See [CONTRIBUTING.md](CONTRIBUTING.md) for tests, coding conventions and reproducibility guidelines.

## Scope and limitations

The repository does **not** provide pretrained GPT-2 weight compatibility, verified large-scale model quality, FSDP, distributed inference, dynamic request batching, paged attention, quantization, speculative decoding or an HTTP serving stack. The bundled CPU examples and tests validate functional behavior, not competitive model performance; CUDA/NCCL paths have not been benchmarked in the development environment.

Influenced by [BuildAnLLM](https://github.com/jammastergirish/BuildAnLLM#resources), [Raschka's *Build a Large Language Model (From Scratch)*](https://www.sebastianraschka.com/llms-from-scratch/), [ARENA](https://learn.arena.education/chapter1_transformer_interp/), [Stanford CS336](https://cs336.stanford.edu/), and [nanochat](https://github.com/karpathy/nanochat). 
