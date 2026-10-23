#!/usr/bin/env bash
# Q2 (verifier audit) and Q3 (landscape) on scale-16m S2 checkpoints, same eval subset as the thinking study.
# PART=final|curve: final = the 250k checkpoint (Q2 at sigma 3 and 0, Q3); curve = Q2 sigma 3 at 50k-200k.
# The 3.36M point is the thinking study's existing s2 105k results (same eval subset).
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$WS/../.." && pwd)"
TS="$REPO/experiments/thinking_study"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export TMPDIR=/mnt/data/quangnx/tmp HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
cd "$REPO"
q2() {  # key step noise tpp_key tpp_step
  local out="$WS/results/q2_verifier_quality/${1}_step${2}_noise${3}"
  $PY "$TS/q2_verifier_quality/run_q2.py" --model "$1" --step "$2" --noise "$3" --tpp "$4" --tpp-step "$5" --out-dir "$out"
  $PY "$TS/q2_verifier_quality/analyze_q2.py" "$out" > "$out/metrics.log"
}
q3() {  # key step
  local out="$WS/results/q3_energy_landscape/${1}_step${2}"
  $PY "$TS/q3_energy_landscape/run_q3.py" --model "$1" --step "$2" --out-dir "$out"
  $PY "$TS/q3_energy_landscape/analyze_q3.py" "$out"
}
case "${PART:?set PART=final|curve}" in
  final)
    for noise in 3 0; do q2 s2_b64_16m 250000 "$noise" tpp_b64_16m 250000; done
    q3 s2_b64_16m 250000 ;;
  curve)
    for step in 50000 100000 150000 200000; do q2 s2_b64_16m "$step" 3 tpp_b64_16m "$step"; done ;;
esac
echo "=== done $(date -Is)"
