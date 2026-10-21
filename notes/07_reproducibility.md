# 07 — A reproducible experiment is more than a checkpoint

## Minimum experiment record

- Corpus identity, document split rule, number of documents, preprocessing,
  tokenizer training split, vocabulary, and token counts per split.
- Architecture hyperparameters, exact seed, optimizer/schedule, precision,
  context length, batch size, accumulation steps and total training tokens.
- Software/hardware versions, accelerator and memory, elapsed time and
  throughput, evaluation method and total validation samples.
- Metrics and generated samples with full decoding hyperparameters.
- Limits and negative results, not merely a best-looking example.

Our `manifest.json` and `run_config.json` record most of these ingredients.
`metrics.jsonl` provides loss, gradient norm, LR, seen tokens, wall-time and
held-out loss/perplexity at evaluation points. For a scientific comparison,
add a model card, data documentation, multiple seeds and matched-cost controls.

## Same-tokenizer ablations

Comparing an attention architecture with one other component changed is
reasonable only when both models see the same training and validation data,
use the same tokenizer, comparable parameter count (or are clearly annotated
as unequal), training steps/tokens, and decoding budget. In particular,
**do not interpret smaller token-level perplexity from one tokenizer as a
better language model than another tokenizer by itself**.

## Data leakage and security

Raw documents are deduplicated prior to shuffling, preventing exact duplicate
documents across splits. Near-duplicates, benchmark contamination, private
strings and source-level licensing issues still require additional checks
before using a real dataset. Downloading arbitrary repositories/scripts or
loading untrusted checkpoint files should not be part of an unattended
experiment. This educational repository makes **no network requests**.

## Reproduction task

1. Run the CPU smoke script, record `metrics.jsonl` and run config.
2. Re-run from a fresh directory, same seed and software; compare checkpoint
   tensors and log values excluding wall time.
3. Change one configuration element and articulate the expected mechanism.
4. Before publishing a result, run three seeds and report variation.
