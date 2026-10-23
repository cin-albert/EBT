#!/usr/bin/env bash
# OOD eval of S2 250k/b64: Q2 at σ=3 (RQ1, self-verification) and Q3 Part A (RQ2, thinking longer) per OOD set.
# Env: GPU_ID (required); N_SEQS (default 1024); DATASETS (default: all four); OUT (default results/ood).
set -euo pipefail
OOD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$OOD/.." && pwd)"
TS="$(cd "$WS/../thinking_study" && pwd)"
REPO="$(cd "$WS/../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export TMPDIR=/mnt/data/quangnx/tmp HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
N_SEQS="${N_SEQS:-1024}"
OUT="${OUT:-$WS/results/ood}"
DATASETS="${DATASETS:-gsm8k squad bigbench_elementary_math_qa bigbench_dyck_languages}"
cd "$REPO"
for data in $DATASETS; do
  echo "=== $data $(date -Is)"
  q2="$OUT/q2_verifier_quality/s2_b64_16m_step250000_${data}_noise3"
  $PY "$TS/q2_verifier_quality/run_q2.py" --model s2_b64_16m --step 250000 --tpp tpp_b64_16m --tpp-step 250000 \
    --noise 3 --data "$data" --n-seqs "$N_SEQS" --out-dir "$q2"
  $PY "$TS/q2_verifier_quality/analyze_q2.py" "$q2" > "$q2/metrics.log"
  q3="$OUT/q3_energy_landscape/s2_b64_16m_step250000_${data}"
  $PY "$TS/q3_energy_landscape/run_q3.py" --model s2_b64_16m --step 250000 --data "$data" --n-seqs-a "$N_SEQS" \
    --skip-part-b --out-dir "$q3"
  $PY "$TS/q3_energy_landscape/analyze_q3.py" "$q3"
done
echo "=== done $(date -Is)"
