#!/usr/bin/env bash
# Full 105k runs, one per "RULE ALPHA" argument, sequential on one GPU; checkpoints every 15k (all kept).
# Usage: GPU_ID=5 ./full.sh "logits 500" "log_probs 5"
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for cfg in "$@"; do
  read -r rule alpha <<< "$cfg"
  name="s2-rule-${rule//_/-}-105k"
  "$WS/run_rule.sh" "$rule" "$alpha" "$name" --save_top_k_ckpts -1 > "$WS/logs/$name.log" 2>&1
done
