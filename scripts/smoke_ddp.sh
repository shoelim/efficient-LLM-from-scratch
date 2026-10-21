#!/usr/bin/env bash
# Two CPU ranks test DDP without leaving data or checkpoints in the repository.
# For NVIDIA GPUs: torchrun --standalone --nproc_per_node=N -m llmfs train ...
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
python -m llmfs prepare --input examples/toy_corpus.txt --output "$workdir/data" --tokenizer char
python -m torch.distributed.run --standalone --nproc-per-node=2 -m llmfs train \
  --data "$workdir/data" --config configs/ddp_smoke.yaml --output "$workdir/run"
SMOKE_CHECKPOINT="$workdir/run/last.pt" python - <<'PY'
import os
from llmfs.training.checkpoint import load_checkpoint
p = load_checkpoint(os.environ['SMOKE_CHECKPOINT'])
assert p['world_size'] == 2
assert len(p['distributed_rng_states']) == 2
assert p['step'] == 3
print('DDP smoke passed: 2 ranks, 3 updates, per-rank RNG saved')
PY
