#!/usr/bin/env bash
# Shared settings: xxs, paper hparams as in ../reproduce_xxs_scaling/common.sh, but DDP on 2 GPUs (global batch 64),
# a 16.1M-doc superset dataset, and the reproduction's fixed 33,320-doc validation set.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$WS/../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface
mkdir -p "$WS/logs" "$WS/runs"

: "${GPU_IDS:?set GPU_IDS to two idle GPUs on one NUMA node, e.g. 4,5}"
for id in ${GPU_IDS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$id")
  if [ "$used" -gt 1000 ] && [ "${ALLOW_BUSY_GPU:-0}" != 1 ]; then echo "GPU $id uses ${used} MiB" >&2; exit 1; fi
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRACKIO_DIR="$WS/runs/trackio"
export TMPDIR=/mnt/data/quangnx/tmp  # / (with /home, /tmp) is shared and filled up once (crashed S2 at step 50,976); short path: AF_UNIX sockets cap at 107 chars
cd "$WS/runs"  # train_model.py writes ./logs/{checkpoints,csv}

COMMON_ARGS=(
  --modality NLP --model_size xxs
  --pretokenize_dataset --tokenizer EleutherAI/gpt-neox-20b
  --context_length 256 --ffn_dim_multiplier 1
  --gpus 2 --distributed_strategy ddp
  --peak_learning_rate 0.0012 --batch_size_per_device 32 --accumulate_grad_batches 1
  --gradient_clip_val 1.0 --weight_decay 0.01 --min_lr_scale 10 --warm_up_steps 10000
  --dataset_name pajama_subset --dataset_dir "$WS/data/full16m"
  --fixed_val_indices "$WS/val_indices_repro_33320.npy" --validation_split_pct 0.0098
  --num_workers 12 --save_top_k_ckpts -1
  --wandb_project ebt_scale16m
  --log_model_archi --log_gradients
  --set_matmul_precision medium
)
