#!/usr/bin/env bash
# Q2 (noise 0; the rule runs train without Langevin noise, see report §2.1) and Q3 for each rule run at 105k;
# same subsets and seeds as the baseline. Q3 Part A covers noise 0 and the run's training noise.
# Usage: GPU_ID=5 [STEP=30000] ./eval.sh s2_rule_log_probs   (STEP defaults to 105000)
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$WS/../../thinking_study"
REPO="$(cd "$WS/../../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
STEP="${STEP:-105000}"
cd "$REPO"
for model in "$@"; do
  for noise in 0; do
    out="$WS/results/q2/${model}_step${STEP}_noise${noise}"
    "$PY" "$TS/q2_verifier_quality/run_q2.py" --model "$model" --step "$STEP" --noise "$noise" --out-dir "$out"
    "$PY" "$TS/q2_verifier_quality/analyze_q2.py" "$out" > "$out/metrics.log"
  done
  out="$WS/results/q3/${model}_step${STEP}"
  "$PY" "$TS/q3_energy_landscape/run_q3.py" --model "$model" --step "$STEP" --out-dir "$out"
  "$PY" "$TS/q3_energy_landscape/analyze_q3.py" "$out"
  "$PY" "$TS/q3_energy_landscape/analyze_q3_extra.py" "$out"
done
