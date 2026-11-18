#!/usr/bin/env bash
# Full-set Table 3 protocol for one run of table3_ema.RUNS (tpp, s2, ema) on one GPU from 0-3.
# Validation = all 33,000 documents of data/val_subset. Sets whose .npz exists are skipped.
# Usage: GPU_ID=0 RUN=ema bash run_table3.sh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"  # sets WS to scale_to_paper_level, PY, REPO, require_gpus
require_gpus "${GPU_ID:?set GPU_ID (one of 0-3)}"
MW="$WS/energy_momentum"
export PYTHONPATH="$REPO:$REPO/experiments/thinking_study/common:$WS:$MW"
export EBT_VAL_SUBSET="${EBT_VAL_SUBSET-$WS/data/val_subset}"
cd "$REPO"
echo "=== ${RUN:?set RUN} $(date -Is)"
"$PY" "$MW/table3_ema.py" score --run "$RUN"
echo "=== done $(date -Is)"
