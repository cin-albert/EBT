"""Check the 16.1M-doc build is an exact superset of the reproduction's 3.4M-doc build and the fixed split is sound."""
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent
SUB = "pajama_subset/EleutherAI_gpt-neox-20b/max_length_257"
OLD = WS.parent / "reproduce_xxs_scaling/data/full" / SUB
NEW = WS / "data/full16m" / SUB
MIN_TRAIN = 250_000 * 64
CHUNK = 200_000


def main() -> int:
    old_ids = np.load(OLD / "input_ids.npy", mmap_mode="r")
    new_ids = np.load(NEW / "input_ids.npy", mmap_mode="r")
    old_len, new_len = np.load(OLD / "lengths.npy"), np.load(NEW / "lengths.npy")
    n_old, n_new = len(old_len), len(new_len)
    val = np.load(WS / "val_indices_repro_33320.npy")
    checks = {
        "lengths prefix equal": bool(np.array_equal(old_len, new_len[:n_old])),
        "ids prefix equal": all(np.array_equal(old_ids[s:min(s + CHUNK, n_old)], new_ids[s:min(s + CHUNK, n_old)]) for s in range(0, n_old, CHUNK)),
        "val indices inside old prefix": bool(val.max() < n_old),
        f"train rows >= {MIN_TRAIN}": n_new - len(val) >= MIN_TRAIN,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"old {n_old:,} docs, new {n_new:,} docs, train {n_new - len(val):,}, val {len(val):,}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
