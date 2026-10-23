#!/usr/bin/env bash
# EBT System 2 xxs at paper scale; MCMC hparams from paper Table D.4 (as ../scale_to_16m_sequences/run_s2.sh).
# Usage: GPU_IDS=0,1,2,3 bash run_s2.sh --run_name s2-paper [extra args]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
require_gpus "${GPU_IDS:?set GPU_IDS, e.g. 0,1,2,3}"
cd "$WS/runs"  # train_model.py writes ./logs/{checkpoints,csv}
exec "$PY" "$REPO/train_model.py" --model_name ebt "${COMMON_ARGS[@]}" \
  --normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise \
  --no_mcmc_detach --mcmc_replay_buffer --truncate_mcmc --langevin_dynamics_noise 3.0 \
  --mcmc_step_size 5 --randomize_mcmc_step_size_scale 2 \
  --mcmc_num_steps 1 --randomize_mcmc_num_steps 2 --randomize_mcmc_num_steps_min 2 \
  "$@"
