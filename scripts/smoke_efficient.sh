#!/usr/bin/env bash
# CPU end-to-end smoke test: all generated artifacts live in a temporary folder.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
python -m llmfs prepare --input examples/toy_corpus.txt --output "$workdir/data" --tokenizer char
python -m llmfs train --data "$workdir/data" --config configs/smoke.yaml --output "$workdir/run"
python -m llmfs export --checkpoint "$workdir/run/last.pt" --output "$workdir/export"
python -m llmfs evaluate --checkpoint "$workdir/run/last.pt" --data "$workdir/data" --batches 2
python -m llmfs generate --checkpoint "$workdir/export" --prompt 'once upon' --max-new-tokens 8 --temperature 0
python scripts/benchmark_llm.py --device cpu --batch 1 --seq 16 --new-tokens 4 --repeats 2 --warmup 1
