"""Write the paper-style validation set: 33,000 documents drawn uniformly from the full build (numpy seed 33).

Usage: python make_val_indices.py <dataset_dir>   (dataset_dir as passed to --dataset_dir, e.g. data/full)
Writes val_indices_paper_33000.npy next to this script, then checks it against the real dataset.
"""
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent
REPO = WS.parents[1]
OUT = WS / "val_indices_paper_33000.npy"


def make_val_indices(n_docs: int, n_val: int = 33_000, seed: int = 33) -> np.ndarray:
    idx = np.random.default_rng(seed).choice(n_docs, n_val, replace=False)
    return np.sort(idx).astype(np.int64)


def main(dataset_dir: str) -> None:
    sys.path.insert(0, str(REPO))
    from data.fixed_val_split import fixed_val_split
    from data.nlp.pajama_subset import RedPajamaSubsetDataset

    hparams = Namespace(execution_mode="pretrain", context_length=256, dataset_dir=dataset_dir,
                        tokenizer="EleutherAI/gpt-neox-20b")
    full = RedPajamaSubsetDataset(hparams)
    np.save(OUT, make_val_indices(len(full)))

    train, val = fixed_val_split(full, str(OUT))
    assert len(train) + len(val) == len(full)
    assert not np.intersect1d(np.asarray(train.indices), np.asarray(val.indices)).size
    print(f"wrote {OUT}: {len(val):,} val / {len(train):,} train of {len(full):,} docs")


if __name__ == "__main__":
    main(sys.argv[1])
