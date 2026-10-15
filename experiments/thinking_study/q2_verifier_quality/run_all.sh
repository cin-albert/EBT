#!/usr/bin/env bash
# Q2 sweep: S1/S2 at 105k and 90k; S2 with the training Langevin noise (3) and without it.
set -euo pipefail
TS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$TS/../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
cd "$REPO"
OUT="$TS/q2_verifier_quality/results"
for cfg in "s2 105000 3" "s2 105000 0" "s1 105000 0" "s2 90000 3" "s1 90000 0"; do
  read -r model step noise <<< "$cfg"
  run="${model}_step${step}_noise${noise}"
  echo "=== $run $(date -Is)"
  $PY "$TS/q2_verifier_quality/run_q2.py" --model "$model" --step "$step" --noise "$noise" --out-dir "$OUT/$run" "$@"
  $PY "$TS/q2_verifier_quality/analyze_q2.py" "$OUT/$run" > "$OUT/$run/metrics.log"
done
echo "=== done $(date -Is)"
