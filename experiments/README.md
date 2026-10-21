# Experiment records

Create one folder per experiment and include:

- `config.yaml` (copied unmodified from the exact run)
- `data_manifest.json` (or a reference to a versioned manifest)
- `environment.txt`: Python, PyTorch, CUDA, GPU, OS, package versions
- `results.md`: purpose, predictions, baselines, train/validation token counts,
  wall time, loss curves, sample generations and limitations
- `metrics.jsonl` copied from `runs/<run-name>/`

Checkpoints belong under `runs/` or a separate artifact store and should
not be committed with code. Never fabricate an experimental result to fill
in a table. A missing result is a missing experiment.
