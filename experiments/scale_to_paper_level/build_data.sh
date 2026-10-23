#!/usr/bin/env bash
# Build the pre-tokenized RedPajama-V2 sample-100B data (first 257 tokens per document, shard order seed 33).
# Usage: bash build_data.sh smoke   -> $WS/data/smoke (20,000 docs)
#        bash build_data.sh full    -> $DATA_DIR (all 8,400 shards), then acceptance checks -> logs/data_stats.json
# CPU only; no GPU is used.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_disk
MODE=${1:?usage: build_data.sh smoke|full}
case "$MODE" in
  smoke) OUT="$WS/data/smoke"; NUM_DOCS=20000; WORKERS=8 ;;
  full) OUT="$DATA_DIR"; NUM_DOCS=80000000; WORKERS=32 ;;  # capacity above any plausible count; stops when tags run out
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
cd "$REPO"
"$PY" -m data.nlp.pajama_subset --dataset_dir "$OUT" --num_docs "$NUM_DOCS" --download_workers "$WORKERS" \
  2>&1 | tee "$WS/logs/build_${MODE}.log"
[ "$MODE" = full ] || exit 0

# Acceptance: fail on > 1% missing shards or a corpus far below the paper's 66M documents.
"$PY" - "$OUT/pajama_subset/EleutherAI_gpt-neox-20b/max_length_257/meta.json" "$WS/logs/data_stats.json" <<'EOF'
import json, sys
meta = json.load(open(sys.argv[1]))
n, missing = meta["num_docs"], len(meta["tags_missing"])
stats = {
    "num_docs": n,
    "ratio_to_paper_66M": round(n / 66e6, 4),
    "num_tags_total": meta["num_tags_total"],
    "num_tags_used": len(meta["tags_used"]),
    "num_tags_missing": missing,
    "tags_missing": meta["tags_missing"],
    "docs_per_language": meta["docs_per_language"],
    "num_short_docs_skipped": meta["num_short_docs_skipped"],
}
json.dump(stats, open(sys.argv[2], "w"), indent=2)
print(json.dumps({k: v for k, v in stats.items() if k != "tags_missing"}, indent=2))
assert missing <= 84, f"{missing} shards missing (> 1% of 8,400)"
assert n >= 50_000_000, f"only {n:,} documents built"
print("acceptance: OK")
EOF
