#!/usr/bin/env bash
# Shared settings for the EBT xxs scaling reproduction (paper Table D.1-D.3, App. D.1.1) on 1 GPU.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface
DATA_ROOT="${DATA_ROOT:-$EXP/data/full}"   # $EXP/data -> /mnt/data/quangnx/research/ebt/data
LOG_DIR="$EXP/logs"
RUNS_DIR="$EXP/runs"
mkdir -p "$LOG_DIR" "$RUNS_DIR"

: "${GPU_ID:?set GPU_ID to an idle GPU index (see nvidia-smi)}"
used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
if [ "$used_mib" -gt 1000 ] && [ "${ALLOW_BUSY_GPU:-0}" != 1 ]; then
  echo "GPU $GPU_ID already uses ${used_mib} MiB; pick an idle GPU or set ALLOW_BUSY_GPU=1" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TRACKIO_DIR="$RUNS_DIR/trackio"

# train_model.py writes ./logs/{checkpoints,csv,...}; running from runs/ keeps every artifact there
cd "$RUNS_DIR"

COMMON_ARGS=(
  --modality NLP --model_size xxs
  --pretokenize_dataset --tokenizer EleutherAI/gpt-neox-20b
  --context_length 256 --ffn_dim_multiplier 1
  --gpus 1
  --peak_learning_rate 0.0012 --batch_size_per_device 32 --accumulate_grad_batches 1
  --gradient_clip_val 1.0 --weight_decay 0.01 --min_lr_scale 10
  --max_steps 105000 --max_scheduling_steps 105000 --warm_up_steps 10000
  --dataset_name pajama_subset --dataset_dir "$DATA_ROOT"
  --num_workers 12 --validation_split_pct 0.0098 --val_check_interval 15000
  --wandb_project ebt_xxs_scaling
  --log_model_archi --log_gradients
  --set_matmul_precision medium
)
