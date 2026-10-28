#!/usr/bin/env bash
# Thinking-study evals of the paper-scale runs, on one GPU from 0-3.
# PART=final: Table 3 checkpoints; per set in DATASETS: Q2 σ=3 (32 candidates, ≤ 1,024 seqs; T++ CE on the same
#             tokens) and Q3 Part A (k = 1..32 at noise 0 and 3, same 1,024 seqs).
# PART=curve: Q2 σ=3 at all 20 S2 checkpoints on each set in DATASETS (default rp_val and Dyck).
# Env: GPU_ID (required), PART, DATASETS, S2_STEP / TPP_STEP (Table 3 steps, default 1000000).
# A run whose metrics.json exists is skipped, so final and curve can share the step-1M outputs.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_gpus "${GPU_ID:?set GPU_ID (one of 0-3)}"
TS="$REPO/experiments/thinking_study"
export PYTHONPATH="$REPO:$TS/common"
S2_STEP="${S2_STEP:-1000000}"
TPP_STEP="${TPP_STEP:-1000000}"  # also the curve's T++ reference: step 800k has two checkpoints (resume), and C4/C5 do not use T++
cd "$REPO"

q2() {  # step data
  local out="$WS/results/q2/s2_paper_step${1}_${2}_noise3"
  [ -f "$out/metrics.json" ] && { echo "skip $out"; return; }
  "$PY" "$TS/q2_verifier_quality/run_q2.py" --model s2_paper --step "$1" --tpp tpp_paper --tpp-step "$TPP_STEP" \
    --noise 3 --data "$2" --n-seqs 1024 --out-dir "$out"
  "$PY" "$TS/q2_verifier_quality/analyze_q2.py" "$out" > "$out/metrics.log"
}
q3() {  # step data
  local out="$WS/results/q3/s2_paper_step${1}_${2}"
  [ -f "$out/metrics.json" ] && { echo "skip $out"; return; }
  "$PY" "$TS/q3_energy_landscape/run_q3.py" --model s2_paper --step "$1" --data "$2" --n-seqs-a 1024 \
    --skip-part-b --out-dir "$out"
  "$PY" "$TS/q3_energy_landscape/analyze_q3.py" "$out"
}

case "${PART:?set PART=final|curve}" in
  final)
    for data in ${DATASETS:-rp_val gsm8k squad bigbench_elementary_math_qa bigbench_dyck_languages}; do
      echo "=== final $data $(date -Is)"; q2 "$S2_STEP" "$data"; q3 "$S2_STEP" "$data"
    done ;;
  curve)
    for data in ${DATASETS:-rp_val bigbench_dyck_languages}; do
      for step in $(seq 50000 50000 1000000); do echo "=== curve $data $step $(date -Is)"; q2 "$step" "$data"; done
    done ;;
  *) echo "unknown PART $PART" >&2; exit 1 ;;
esac
echo "=== done $(date -Is)"
