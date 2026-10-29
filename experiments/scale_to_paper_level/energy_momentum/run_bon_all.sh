#!/usr/bin/env bash
# BoN candidates on one GPU from 0-3: run_bon.py for every arm in ARMS x k in KS x set in DATASETS.
# Env: GPU_ID, ARMS, KS (required), DATASETS (default: rp_val + 4 OOD sets). A run whose config.json exists is skipped.
# Usage: GPU_ID=0 ARMS="gd_a5 ema_m0.5" KS="3 5" bash run_bon_all.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
require_gpus "${GPU_ID:?set GPU_ID (one of 0-3)}"
MW="$WS/energy_momentum"
export PYTHONPATH="$REPO:$REPO/experiments/thinking_study/common:$MW"
export EBT_VAL_SUBSET="${EBT_VAL_SUBSET-$WS/data/val_subset}"
cd "$REPO"
for arm in ${ARMS:?set ARMS}; do
  for k in ${KS:?set KS}; do
    for data in ${DATASETS:-rp_val gsm8k squad bigbench_elementary_math_qa bigbench_dyck_languages}; do
      out="$MW/results/bon/${arm}_k${k}_${data}"
      [ -f "$out/config.json" ] && { echo "skip $out"; continue; }
      echo "=== $arm k=$k $data $(date -Is)"
      "$PY" "$MW/run_bon.py" --arm "$arm" --k "$k" --data "$data" --out-dir "$out" | tail -1
    done
  done
done
echo "=== done $(date -Is)"
