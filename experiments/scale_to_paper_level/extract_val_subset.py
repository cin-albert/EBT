"""Copy the 33,000 validation documents of the full build into a small standalone pajama_subset (~17 MB).

The full build (data/full, ~195 GB with the raw cache) is too large to keep. The copy has the same on-disk
format, so RedPajamaSubsetDataset reads it directly; with --dataset_dir val_subset and --fixed_val_indices
val_subset/val_indices.npy (= 0..32,999) the validation split yields the same documents in the same order
as the full build with val_indices_paper_33000.npy. The train split of the copy is empty.

Usage: python extract_val_subset.py [src_dataset_dir] [val_indices] [out_dir]
       (defaults: data/full, val_indices_paper_33000.npy, val_subset; all relative to this script)
"""
import json
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS.parents[1]))
from data.nlp.pajama_subset import SubsetWriter, subset_dir  # noqa: E402

TOKENIZER = "EleutherAI/gpt-neox-20b"
MAX_LENGTH = 257  # context_length 256 + 1


def extract_val_subset(src_dataset_dir: str, val_indices_path: str, out_dir: str,
                       tokenizer: str = TOKENIZER, max_length: int = MAX_LENGTH) -> Path:
    """Write the documents at val_indices (in file order) as a pajama_subset build under out_dir; return its root."""
    src = Path(subset_dir(src_dataset_dir, tokenizer, max_length))
    meta = json.loads((src / "meta.json").read_text())
    lengths = np.load(src / "lengths.npy")
    input_ids = np.load(src / "input_ids.npy", mmap_mode="r")
    val_idx = np.load(val_indices_path)

    writer = SubsetWriter(subset_dir(out_dir, tokenizer, max_length), len(val_idx), max_length)
    for i in val_idx:
        writer.add(input_ids[i, : lengths[i]].tolist())
    writer.finalize(dict(meta, source_dataset_dir=str(Path(src_dataset_dir).resolve()),
                         source_num_docs=meta["num_docs"], source_val_indices=str(Path(val_indices_path).resolve())))

    out = Path(out_dir)
    np.save(out / "val_indices.npy", np.arange(len(val_idx), dtype=np.int64))
    np.save(out / "source_indices.npy", val_idx)  # row j of the copy is document source_indices[j] of the full build
    return Path(writer.out_dir)


if __name__ == "__main__":
    args = sys.argv[1:] + [None] * 3
    root = extract_val_subset(args[0] or str(WS / "data/full"), args[1] or str(WS / "val_indices_paper_33000.npy"),
                              args[2] or str(WS / "val_subset"))
    print(f"wrote {root}")
