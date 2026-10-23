#!/usr/bin/env bash
# Transformer++ xxs; caller passes --run_name, --max_steps, --val_check_interval (max_scheduling_steps follows max_steps).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" "$REPO/train_model.py" --model_name baseline_transformer "${COMMON_ARGS[@]}" "$@"
