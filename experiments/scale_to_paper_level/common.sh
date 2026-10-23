#!/usr/bin/env bash
# Shared settings for the paper-scale runs: xxs, global batch 128 (4 x 32) on GPUs 0-3 only,
# the full sample-100B build, and a new random 33,000-doc validation set.
# Launch scripts call require_gpus "$GPU_IDS" (which also defines COMMON_ARGS) before anything else.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$WS/../.." && pwd)"
PY="${PY:-/home/quangnx/miniconda3/envs/ebt312/bin/python}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false  # collate runs in forked DataLoader workers
export TMPDIR="${TMPDIR_EBT:-/tmp/ebt_paper}"  # short path: AF_UNIX socket paths cap at 107 chars
DATA_DIR="${DATA_DIR:-$WS/data/full}"
ALLOWED_GPUS=(0 1 2 3)  # GPUs 4 and 5 belong to other users
MIN_FREE_GB=500
mkdir -p "$WS/logs" "$WS/runs" "$TMPDIR"

check_disk() {
  local free_gb
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then echo "only ${free_gb} GB free on /, need ${MIN_FREE_GB}" >&2; exit 1; fi
}

require_gpus() {  # $1 = comma-separated GPU ids; exports CUDA_VISIBLE_DEVICES, sets NUM_GPUS and COMMON_ARGS
  local ids=${1:?set GPU_IDS, e.g. 0,1,2,3} id used
  for id in ${ids//,/ }; do
    if [[ ! " ${ALLOWED_GPUS[*]} " == *" $id "* ]]; then echo "GPU $id not allowed; use only ${ALLOWED_GPUS[*]}" >&2; exit 1; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$id")
    if [ "$used" -gt 1000 ] && [ "${ALLOW_BUSY_GPU:-0}" != 1 ]; then echo "GPU $id uses ${used} MiB" >&2; exit 1; fi
  done
  check_disk
  export CUDA_VISIBLE_DEVICES="$ids"
  NUM_GPUS=$(echo "${ids//,/ }" | wc -w)
  COMMON_ARGS=(
    --modality NLP --model_size xxs
    --pretokenize_dataset --tokenizer EleutherAI/gpt-neox-20b
    --context_length 256 --ffn_dim_multiplier 1
    --gpus "$NUM_GPUS" --distributed_strategy ddp
    --peak_learning_rate 0.0012 --batch_size_per_device 32 --accumulate_grad_batches 1
    --gradient_clip_val 1.0 --weight_decay 0.01 --min_lr_scale 10 --warm_up_steps 10000
    --max_steps 1000000 --max_scheduling_steps 1000000
    --dataset_name pajama_subset --dataset_dir "$DATA_DIR"
    --fixed_val_indices "$WS/val_indices_paper_33000.npy"
    --val_check_interval 50000 --check_val_every_n_epoch 0
    --num_workers 12 --dataloader_mp_context forkserver --save_top_k_ckpts -1
    --wandb_project ebt_scale_paper
    --log_model_archi --log_gradients
    --set_matmul_precision medium
  )
  export TRACKIO_DIR="$WS/runs/trackio"
}
