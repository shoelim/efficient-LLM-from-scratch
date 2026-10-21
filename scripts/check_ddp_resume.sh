#!/usr/bin/env bash
# Strong CPU check: a two-rank DDP run resumed after step 2 must match 4 steps uninterrupted.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
python -m llmfs prepare --input examples/toy_corpus.txt --output "$workdir/data" --tokenizer char >/dev/null
SMOKE_WORKDIR="$workdir" python - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ['SMOKE_WORKDIR'])
config = yaml.safe_load(Path('configs/ddp_smoke.yaml').read_text())
config['train']['steps'] = 4
config['train']['save_interval'] = 2
(root / 'config.yaml').write_text(yaml.safe_dump(config))
PY
runner=(python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m llmfs train)
args=(--data "$workdir/data" --config "$workdir/config.yaml")
"${runner[@]}" "${args[@]}" --output "$workdir/full" >"$workdir/full.log"
"${runner[@]}" "${args[@]}" --output "$workdir/partial" --stop-after-step 2 >"$workdir/partial.log"
"${runner[@]}" "${args[@]}" --output "$workdir/partial" \
  --resume "$workdir/partial/last.pt" >"$workdir/resumed.log"
SMOKE_WORKDIR="$workdir" python - <<'PY'
import os
from pathlib import Path
import torch
from llmfs.training.checkpoint import load_checkpoint
root = Path(os.environ['SMOKE_WORKDIR'])
full = load_checkpoint(root / 'full' / 'last.pt')
resumed = load_checkpoint(root / 'partial' / 'last.pt')
assert full['step'] == resumed['step'] == 4
assert full['world_size'] == resumed['world_size'] == 2
for name, value in full['model'].items():
    assert torch.equal(value, resumed['model'][name]), name
for i, state in full['optimizer']['state'].items():
    for k, v in state.items():
        w = resumed['optimizer']['state'][i][k]
        assert torch.equal(v, w) if torch.is_tensor(v) else v == w
for rank in range(2):
    assert torch.equal(full['distributed_rng_states'][rank]['batch'],
                       resumed['distributed_rng_states'][rank]['batch'])
print('DDP exact-resume parity verified: model, optimizer and batch RNG on both ranks')
PY
