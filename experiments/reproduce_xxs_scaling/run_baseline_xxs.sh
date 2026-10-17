#!/usr/bin/env bash
# Transformer++ baseline with the same data, size and optimizer settings.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" "$REPO/train_model.py" \
  --run_name baseline-xxs-scaling-105k --model_name baseline_transformer \
  "${COMMON_ARGS[@]}" \
  "$@"
