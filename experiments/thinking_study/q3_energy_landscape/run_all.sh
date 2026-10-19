#!/usr/bin/env bash
# Q3: landscape and update-map properties, S2 (primary) then S1, both at 105k.
set -euo pipefail
TS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$TS/../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
cd "$REPO"
for model in s2 s1; do
  echo "=== $model $(date -Is)"
  $PY "$TS/q3_energy_landscape/run_q3.py" --model "$model" --out-dir "$TS/q3_energy_landscape/results/${model}_step105000" --reuse-part-a "$@"
done
echo "=== done $(date -Is)"
