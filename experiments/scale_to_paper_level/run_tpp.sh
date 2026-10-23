#!/usr/bin/env bash
# Transformer++ xxs at paper scale (4 x 32, 1M steps). Usage: GPU_IDS=0,1,2,3 bash run_tpp.sh --run_name tpp-paper [extra args]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_gpus "${GPU_IDS:?set GPU_IDS, e.g. 0,1,2,3}"
cd "$WS/runs"  # train_model.py writes ./logs/{checkpoints,csv}
exec "$PY" "$REPO/train_model.py" --model_name baseline_transformer "${COMMON_ARGS[@]}" "$@"
