#!/usr/bin/env bash
# S2 xxs with a given MCMC update rule and step size; everything else as the probs baseline.
# Usage: GPU_ID=5 ./run_rule.sh RULE ALPHA RUN_NAME [extra train_model.py args]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rule="$1" alpha="$2" name="$3"
shift 3
exec "$PY" "$REPO/train_model.py" --run_name "$name" "${COMMON_ARGS[@]}" "${S2_ARGS[@]}" \
  --mcmc_update_rule "$rule" --mcmc_step_size "$alpha" "$@"
