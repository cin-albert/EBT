import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.fixed_val_split import fixed_val_split  # noqa: E402
from data.nlp.pajama_subset import RedPajamaSubsetDataset, SubsetWriter, subset_dir  # noqa: E402

from extract_val_subset import extract_val_subset  # noqa: E402

TOK = "EleutherAI/gpt-neox-20b"


def tiny_build(dataset_dir: Path, n_docs: int = 50, max_length: int = 257) -> None:
    """A pajama_subset build with distinct token ids and lengths per document."""
    rng = np.random.default_rng(0)
    w = SubsetWriter(subset_dir(str(dataset_dir), TOK, max_length), n_docs, max_length)
    for i in range(n_docs):
        w.add(rng.integers(1, 50_000, size=int(rng.integers(5, max_length + 1))).tolist())
    w.finalize({"tokenizer": TOK, "seed": 33})


def val_items(dataset_dir: Path, indices: Path) -> list[dict]:
    hp = Namespace(execution_mode="pretrain", context_length=256, dataset_dir=str(dataset_dir), tokenizer=TOK)
    val = fixed_val_split(RedPajamaSubsetDataset(hp), str(indices))[1]
    return [val[i] for i in range(len(val))]


def test_extracted_subset_gives_the_same_validation_items_in_order(tmp_path):
    tiny_build(tmp_path / "full")
    np.save(tmp_path / "val.npy", np.array([41, 3, 17, 0, 49], dtype=np.int64))  # unsorted: order must be kept
    out = tmp_path / "val_subset"
    extract_val_subset(str(tmp_path / "full"), str(tmp_path / "val.npy"), str(out))
    assert val_items(out, out / "val_indices.npy") == val_items(tmp_path / "full", tmp_path / "val.npy")


def test_extracted_meta_records_the_source(tmp_path):
    tiny_build(tmp_path / "full")
    np.save(tmp_path / "val.npy", np.array([2, 7], dtype=np.int64))
    out = tmp_path / "val_subset"
    extract_val_subset(str(tmp_path / "full"), str(tmp_path / "val.npy"), str(out))
    meta = json.loads((Path(subset_dir(str(out), TOK, 257)) / "meta.json").read_text())
    assert meta["num_docs"] == 2 and meta["source_num_docs"] == 50
    assert np.array_equal(np.load(out / "source_indices.npy"), [2, 7])
