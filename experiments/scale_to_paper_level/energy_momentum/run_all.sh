#!/usr/bin/env bash
# Momentum grid on one GPU from 0-3: run_momentum.py for every arm in ARMS x set in DATASETS.
# Env: GPU_ID (required), ARMS (space-separated arm names, required), DATASETS (default: rp_val + 4 OOD sets).
# A run whose config.json exists is skipped.
# Usage: GPU_ID=0 ARMS="gd_a5 gd_a7.5" bash run_all.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"  # sets WS to scale_to_paper_level, PY, REPO, require_gpus
require_gpus "${GPU_ID:?set GPU_ID (one of 0-3)}"
MW="$WS/energy_momentum"
export PYTHONPATH="$REPO:$REPO/experiments/thinking_study/common:$MW"
export EBT_VAL_SUBSET="${EBT_VAL_SUBSET-$WS/data/val_subset}"
cd "$REPO"
for arm in ${ARMS:?set ARMS}; do
  for data in ${DATASETS:-rp_val gsm8k squad bigbench_elementary_math_qa bigbench_dyck_languages}; do
    out="$MW/results/single/${arm}_${data}"
    [ -f "$out/config.json" ] && { echo "skip $out"; continue; }
    echo "=== $arm $data $(date -Is)"
    "$PY" "$MW/run_momentum.py" --arm "$arm" --data "$data" --out-dir "$out"
  done
done
echo "=== done $(date -Is)"
