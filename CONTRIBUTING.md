# Contributing

Thank you for helping improve the LLM-from-scratch implementation. This project prioritizes **correctness, clarity, and reproducibility** over complexity.

## Development workflow

1. Create a virtual environment and run `python -m pip install -e '.[dev]'`.
2. Make a focused change in `src/llmfs/`, `scripts/`, or the study materials.
3. Add or update a unit test for behavior changes. Numerical optimization changes should include a reference-implementation comparison (forward and backward where applicable).
4. Run `python -m pytest -q`, `python -m ruff check src tests`, and any relevant smoke or benchmark scripts.
5. In a pull request, describe the motivation, mathematics or invariants, testing environment, and limitations. For performance claims, report matched hardware, batch size, sequence length, dtype, model settings, warmup, and measurements.

## Design guidelines

- Keep components small, explicit, and easy to inspect. Document non-obvious tensor shapes.
- Keep reference implementations available where they help explain or verify optimized code.
- Do not introduce heavyweight dependencies or pretrained weights for a change that can be expressed using PyTorch and the existing package.
- Prefer reproducible fixtures and small CPU tests. GPU-only paths need guarded tests and a documented fallback.
- Never commit generated checkpoints, token arrays, training runs, API keys, private datasets, or local benchmark logs.
- Preserve exact causal masking, cache correctness, label alignment, and resume invariants when editing performance-sensitive paths.

## Reproducible experiments

Place configs and human-written experiment summaries under `experiments/` when useful. Generated outputs belong in the ignored `runs/` and `data/` directories. Record seed, dataset provenance, environment, evaluation protocol, and uncertainty; don't substitute toy-smoke outputs for model-quality benchmarks.

This repository includes an MIT license for its own source. New contributions should be compatible with that license; any external data or code needs an explicit provenance and licensing check.
