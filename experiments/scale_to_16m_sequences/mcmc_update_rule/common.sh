#!/usr/bin/env bash
# Baseline settings (../../reproduce_xxs_scaling/common.sh: xxs, 1 GPU, batch 32, 105k steps, 3.4M-doc data, seed 33)
# with this workspace's runs/ and logs/, plus the S2 flags of ../../reproduce_xxs_scaling/run_ebt_s2_xxs.sh.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WS/../../reproduce_xxs_scaling/common.sh"  # sets PY, REPO, COMMON_ARGS, CUDA_VISIBLE_DEVICES; checks GPU_ID is idle
mkdir -p "$WS/runs" "$WS/logs"
export TRACKIO_DIR="$WS/runs/trackio"
cd "$WS/runs"  # train_model.py writes ./logs/{checkpoints,csv}

S2_ARGS=(
  --model_name ebt
  --normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise
  --no_mcmc_detach --mcmc_replay_buffer --truncate_mcmc --langevin_dynamics_noise 3.0
  --randomize_mcmc_step_size_scale 2
  --mcmc_num_steps 1 --randomize_mcmc_num_steps 2 --randomize_mcmc_num_steps_min 2
  --wandb_project ebt_mcmc_update_rule
)
