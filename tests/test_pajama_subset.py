import json
import os

import numpy as np
import pytest

from conftest import TOKENIZER_NAME, write_shard
from data.nlp.pajama_subset import (
    PAD_ID,
    all_tags,
    build_subset,
    download_documents,
    parse_hf_script_constants,
    tag_language,
    tokenize_batch,
)

LONG_TEXT = " ".join(f"word{i} is followed by more text." for i in range(3000))


def test_parse_hf_script_constants():
    source = '_LANGUAGES = ("en", "de")\nX = 1\n_CC_SNAPSHOT_IDS = (\n    "2014-15",\n    "2023-14",\n)\n'
    assert parse_hf_script_constants(source) == (["en", "de"], ["2014-15", "2023-14"])


def test_all_tags_match_sample_100b_layout():
    tags = all_tags(["en", "de"], ["2014-15", "2023-14"])
    assert len(tags) == 2 * 2 * 2 * 10
    assert tags[0] == "2014-15/0000/en_head"
    assert "2023-14/0009/de_middle" in tags
    assert tag_language("2023-14/0009/de_middle") == "de"


def test_tokenize_batch_matches_original_pipeline(tokenizer):
    original = tokenizer(LONG_TEXT, padding=True, truncation=True, max_length=257)["input_ids"]
    assert tokenize_batch(tokenizer, [LONG_TEXT], 257) == [original]
    assert len(original) == 257


def _build(tmp_path, tokenizer, shards, num_docs, max_length=17):
    paths = {tag: (write_shard(str(tmp_path / "raw"), tag, texts) if texts is not None else None) for tag, texts in shards.items()}
    out_dir = str(tmp_path / "out")
    meta = build_subset(out_dir, num_docs, max_length, tokenizer, TOKENIZER_NAME, list(shards), paths.get, seed=33, download_workers=2)
    return out_dir, meta


def test_build_skips_missing_shards_and_short_docs(tmp_path, tokenizer):
    shards = {
        "2014-15/0000/en_head": ["hello world, this is a document", "", "x", LONG_TEXT],
        "2014-15/0001/en_head": None,  # 404
        "2014-15/0000/de_head": ["guten tag, das ist ein dokument"],
    }
    out_dir, meta = _build(tmp_path, tokenizer, shards, num_docs=100)
    assert meta["num_docs"] == 3
    assert meta["num_short_docs_skipped"] == 2  # "" and "x" leave no next-token target
    assert meta["tags_missing"] == ["2014-15/0001/en_head"]
    assert meta["docs_per_language"] == {"en": 2, "de": 1}
    lengths = np.load(os.path.join(out_dir, "lengths.npy"))
    assert lengths.shape == (3,) and lengths.min() >= 2 and lengths.max() == 17
    with open(os.path.join(out_dir, "meta.json")) as f:
        assert json.load(f)["num_docs"] == 3


def test_build_stops_at_capacity_and_pads(tmp_path, tokenizer):
    shards = {"2014-15/0000/en_head": [f"short document number {i}" for i in range(10)]}
    out_dir, meta = _build(tmp_path, tokenizer, shards, num_docs=4)
    ids = np.load(os.path.join(out_dir, "input_ids.npy"))
    lengths = np.load(os.path.join(out_dir, "lengths.npy"))
    assert meta["num_docs"] == 4 and ids.shape == (4, 17) and ids.dtype == np.uint16
    assert (ids[0, lengths[0]:] == PAD_ID).all()


class _Resp:
    def __init__(self, status, body=b""):
        self.status_code, self._body = status, body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size):
        yield self._body


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, stream, timeout):
        self.calls += 1
        return self.responses.pop(0)


def test_download_returns_none_on_404(tmp_path):
    assert download_documents("2014-15/0000/en_head", str(tmp_path), _Session([_Resp(404)])) is None


def test_download_retries_transient_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("data.nlp.pajama_subset.time.sleep", lambda s: None)
    session = _Session([_Resp(503), _Resp(200, b"data")])
    path = download_documents("2014-15/0000/en_head", str(tmp_path), session)
    with open(path, "rb") as f:
        assert session.calls == 2 and f.read() == b"data"


def test_download_reuses_cache(tmp_path):
    cached = write_shard(str(tmp_path), "2014-15/0000/en_head", ["a"])
    assert download_documents("2014-15/0000/en_head", str(tmp_path), session=None) == cached


from types import SimpleNamespace

from data.nlp.collator import NLP_HF_Collator


def _subset_hparams(tmp_path, context_length=16):
    return SimpleNamespace(execution_mode="pretrain", context_length=context_length, dataset_dir=str(tmp_path), tokenizer=TOKENIZER_NAME, mcmc_replay_buffer=False, pretokenize_dataset=True)


def _build_in_place(tmp_path, tokenizer, texts, max_length=17):
    from data.nlp.pajama_subset import subset_dir
    tag = "2014-15/0000/en_head"
    path = write_shard(str(tmp_path / "raw"), tag, texts)
    out_dir = subset_dir(str(tmp_path), TOKENIZER_NAME, max_length)
    return build_subset(out_dir, 100, max_length, tokenizer, TOKENIZER_NAME, [tag], {tag: path}.get, seed=33, download_workers=1)


def test_dataset_returns_unpadded_items(tmp_path, tokenizer):
    from data.nlp.pajama_subset import RedPajamaSubsetDataset
    _build_in_place(tmp_path, tokenizer, ["a short doc here", LONG_TEXT])
    ds = RedPajamaSubsetDataset(_subset_hparams(tmp_path))
    assert len(ds) == 2
    item = ds[0]
    assert item["input_ids"] == tokenizer("a short doc here")["input_ids"]
    assert item["attention_mask"] == [1] * len(item["input_ids"])
    assert len(ds[1]["input_ids"]) == 17


def test_collator_pads_subset_items(tmp_path, tokenizer):
    from data.nlp.pajama_subset import RedPajamaSubsetDataset
    _build_in_place(tmp_path, tokenizer, ["a short doc here", LONG_TEXT])
    hparams = _subset_hparams(tmp_path)
    ds = RedPajamaSubsetDataset(hparams)
    batch = NLP_HF_Collator(hparams)([ds[0], ds[1]])
    assert tuple(batch["input_ids"].shape) == (2, 17)
    assert batch["input_ids"][0, -1].item() == PAD_ID


def test_dataset_refuses_subset_built_for_other_context_length(tmp_path, tokenizer):
    from data.nlp.pajama_subset import RedPajamaSubsetDataset
    _build_in_place(tmp_path, tokenizer, ["a short doc here"], max_length=17)
    with pytest.raises(FileNotFoundError, match="python -m data.nlp.pajama_subset"):
        RedPajamaSubsetDataset(_subset_hparams(tmp_path, context_length=256))


def test_dataset_refuses_inference_mode(tmp_path):
    from data.nlp.pajama_subset import RedPajamaSubsetDataset
    hparams = _subset_hparams(tmp_path)
    hparams.execution_mode = "inference"
    with pytest.raises(ValueError, match="pretrain"):
        RedPajamaSubsetDataset(hparams)
