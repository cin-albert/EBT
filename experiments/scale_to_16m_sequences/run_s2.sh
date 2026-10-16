#!/usr/bin/env bash
# EBT System 2, MCMC hparams from paper Table D.4 (identical to ../reproduce_xxs_scaling/run_ebt_s2_xxs.sh).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" "$REPO/train_model.py" --model_name ebt "${COMMON_ARGS[@]}" \
  --normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise \
  --no_mcmc_detach --mcmc_replay_buffer --truncate_mcmc --langevin_dynamics_noise 3.0 \
  --mcmc_step_size 5 --randomize_mcmc_step_size_scale 2 \
  --mcmc_num_steps 1 --randomize_mcmc_num_steps 2 --randomize_mcmc_num_steps_min 2 \
  "$@"
