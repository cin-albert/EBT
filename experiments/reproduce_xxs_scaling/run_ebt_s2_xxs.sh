#!/usr/bin/env bash
# EBT System 2, MCMC hparams from paper Table D.4 (NLP column); batch, steps and data match the S1 run (see report §8).
# Not the post-paper ebt_s2.sh values (alpha 0.5, 2-4 steps, --norm_pred, --scale_alpha_with_energy, FineWeb).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" "$REPO/train_model.py" \
  --run_name ebt-s2-xxs-scaling-105k --model_name ebt \
  "${COMMON_ARGS[@]}" \
  --normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise \
  --no_mcmc_detach --mcmc_replay_buffer --truncate_mcmc --langevin_dynamics_noise 3.0 \
  --mcmc_step_size 5 --randomize_mcmc_step_size_scale 2 \
  --mcmc_num_steps 1 --randomize_mcmc_num_steps 2 --randomize_mcmc_num_steps_min 2 \
  "$@"
