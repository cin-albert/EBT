#!/usr/bin/env bash
# Gate (probs, 150 steps, must match the baseline log), then the three 2k-step α pilots; one GPU, sequential.
# Extra args go to every pilot run and TAG is appended to their names; SKIP_GATE=1 skips the gate.
# Usage: GPU_ID=5 ./pilot.sh                                          (first round)
#        GPU_ID=5 SKIP_GATE=1 TAG=-n0 ./pilot.sh --langevin_dynamics_noise 0   (noise-0 round)
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
FAST=(--limit_val_batches 0 --save_top_k_ckpts 0)
TAG="${TAG:-}"
csv() { echo "$WS/runs/logs/csv/$1/version_0/metrics.csv"; }
mkdir -p "$WS/logs"

if [ "${SKIP_GATE:-0}" != 1 ]; then
  "$WS/run_rule.sh" probs 5 gate-probs-150 --max_steps 150 "${FAST[@]}" > "$WS/logs/gate-probs-150.log" 2>&1
  "$PY" "$WS/check_runs.py" gate "$(csv gate-probs-150)" | tee "$WS/logs/gate.json"
fi

names=()
for cfg in "logits 5" "logits 500" "log_probs 5"; do
  read -r rule alpha <<< "$cfg"
  name="pilot-${rule//_/-}-a${alpha}${TAG}"
  names+=("$(csv "$name")")
  "$WS/run_rule.sh" "$rule" "$alpha" "$name" --max_steps 2000 "${FAST[@]}" "$@" > "$WS/logs/$name.log" 2>&1 \
    || echo "$name exited with status $?"
done
"$PY" "$WS/check_runs.py" pilot "${names[@]}" | tee "$WS/logs/pilot${TAG}.json"
