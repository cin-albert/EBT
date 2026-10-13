#!/usr/bin/env bash
# EBT System 1, hparams from ebt_s1.sh before commit 19420cb (= paper Table D.3, NLP column).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" "$REPO/train_model.py" \
  --run_name ebt-xxs-scaling-105k --model_name ebt \
  "${COMMON_ARGS[@]}" \
  --normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise \
  --mcmc_step_size_learnable --mcmc_step_size 500 --mcmc_step_size_lr_multiplier 1500 --mcmc_num_steps 2 \
  "$@"
