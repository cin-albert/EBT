#!/usr/bin/env bash
# Sequential DDP queue on GPU_IDS: 250k x 64 = 16M seqs per run, validate (and checkpoint) every 25k.
# T++ first: it is ~5x cheaper and checks the pipeline end to end before the long S2 run.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GPU_IDS:?set GPU_IDS}"
run() {  # model run_name max_steps val_interval [extra]
  local model=$1 name=$2 steps=$3 every=$4; shift 4
  echo "=== $name start $(date -Is)"
  GPU_IDS=$GPU_IDS bash "$WS/run_$model.sh" --run_name "$name" --max_steps "$steps" --val_check_interval "$every" "$@" \
    > "$WS/logs/$name.log" 2>&1
  echo "=== $name end $(date -Is)"
}
run tpp tpp-b64-16m 250000 25000
run s2 s2-b64-16m 250000 25000
echo "=== all done $(date -Is)"
