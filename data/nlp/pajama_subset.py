"""Shard-sampled subset of RedPajama-Data-V2 'sample-100B', pre-tokenized into a uint16 memmap.

The paper's 'pajama' pipeline (data/nlp/pajama_dataloader.py) loads all of sample-100B
(5 languages x 84 snapshots x head/middle x 10 shards, plus quality signals) and keeps only the
first context_length+1 tokens of every document. The xxs scaling run reads 105k steps x 32 docs,
so this module samples whole document shards uniformly from the same tag set and tokenizes them
with the same tokenizer call.
"""
import argparse
import ast
import gzip
import json
import os
import random
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import requests
from torch.utils.data import Dataset

HF_SCRIPT_URL = "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/raw/main/RedPajama-Data-V2.py"
URL_BASE = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
PARTITIONS = ("head", "middle")  # sample-100B uses partition "head_middle"
NUM_SHARDS = 10  # _SUBSAMPLE_FILE_COUNTS["sample-100B"]
MAX_CHARS = 8000  # far more text than 257 tokens need; avoids tokenizing whole documents
PAD_ID = 0  # gpt-neox eos, which NLP_HF_Collator uses as pad
TOKENIZE_BATCH = 1024


def subset_dir(dataset_dir, tokenizer_name, max_length):
    return os.path.join(dataset_dir, "pajama_subset", tokenizer_name.replace("/", "_"), f"max_length_{max_length}")


def parse_hf_script_constants(source):
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("_LANGUAGES", "_CC_SNAPSHOT_IDS"):
                found[node.targets[0].id] = list(ast.literal_eval(node.value))
    return found["_LANGUAGES"], found["_CC_SNAPSHOT_IDS"]


def all_tags(languages, snapshots):
    return [
        f"{snapshot}/{n:04d}/{lang}_{part}"
        for lang in languages
        for snapshot in snapshots
        for part in PARTITIONS
        for n in range(NUM_SHARDS)
    ]


def tag_language(tag):
    return tag.rsplit("/", 1)[1].split("_")[0]


def download_documents(tag, cache_dir, session, retries=5):
    path = os.path.join(cache_dir, "documents", f"{tag}.json.gz")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"{URL_BASE}/documents/{tag}.json.gz"
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, timeout=60) as resp:
                if resp.status_code == 404:  # listed as missing in the HF loading script
                    return None
                resp.raise_for_status()
                tmp_path = path + ".part"
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                os.replace(tmp_path, path)
                return path
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def iter_raw_contents(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["raw_content"]


def tokenize_batch(tokenizer, texts, max_length):
    # Same tokens as RedPajamaDataset.tokenization (per-document call, where padding=True is a no-op).
    # No padding here: on a batch it would pad every document to the longest one.
    return tokenizer([t[:MAX_CHARS] for t in texts], truncation=True, max_length=max_length)["input_ids"]


class SubsetWriter:
    def __init__(self, out_dir, capacity, max_length):
        os.makedirs(out_dir, exist_ok=True)
        meta_path = os.path.join(out_dir, "meta.json")
        if os.path.exists(meta_path):
            os.remove(meta_path)  # a rebuild that crashes must not leave an old meta pointing at a half-written memmap
        self.out_dir, self.capacity, self.max_length = out_dir, capacity, max_length
        self.input_ids = np.lib.format.open_memmap(
            os.path.join(out_dir, "input_ids.npy"), mode="w+", dtype=np.uint16, shape=(capacity, max_length)
        )
        self.lengths = np.zeros(capacity, dtype=np.int16)
        self.count = 0

    @property
    def full(self):
        return self.count >= self.capacity

    def add(self, ids):
        if max(ids) > np.iinfo(np.uint16).max:
            raise ValueError(f"token id {max(ids)} does not fit in uint16")
        row = self.input_ids[self.count]
        row[:] = PAD_ID
        row[: len(ids)] = ids
        self.lengths[self.count] = len(ids)
        self.count += 1

    def finalize(self, meta):
        self.input_ids.flush()
        np.save(os.path.join(self.out_dir, "lengths.npy"), self.lengths[: self.count])
        meta = dict(meta, num_docs=self.count, capacity=self.capacity, max_length=self.max_length)
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return meta


def build_subset(out_dir, num_docs, max_length, tokenizer, tokenizer_name, tags, fetch, seed, download_workers):
    order = list(tags)
    random.Random(seed).shuffle(order)
    writer = SubsetWriter(out_dir, num_docs, max_length)
    used, missing, lang_docs = [], [], Counter()
    n_short = 0

    def consume(texts, lang):
        nonlocal n_short
        if not texts:
            return
        for ids in tokenize_batch(tokenizer, texts, max_length):
            if writer.full:
                return
            if len(ids) < 2:  # needs at least one input token and one next-token target
                n_short += 1
                continue
            writer.add(ids)
            lang_docs[lang] += 1

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        tag_iter, pending = iter(order), deque()

        def submit_next():
            tag = next(tag_iter, None)
            if tag is not None:
                pending.append((tag, pool.submit(fetch, tag)))

        for _ in range(2 * download_workers):
            submit_next()
        while pending and not writer.full:
            tag, future = pending.popleft()
            path = future.result()
            submit_next()
            if path is None:
                missing.append(tag)
                continue
            used.append(tag)
            lang, texts = tag_language(tag), []
            for text in iter_raw_contents(path):
                texts.append(text)
                if len(texts) == TOKENIZE_BATCH:
                    consume(texts, lang)
                    texts = []
                    if writer.full:
                        break
            consume(texts, lang)
            print(f"[pajama_subset] {writer.count:,}/{num_docs:,} docs after {len(used)} shards", flush=True)
        for _, future in pending:
            future.cancel()

    return writer.finalize({
        "source": "togethercomputer/RedPajama-Data-V2 sample-100B tag set (documents only)",
        "tokenizer": tokenizer_name,
        "seed": seed,
        "max_chars": MAX_CHARS,
        "num_tags_total": len(order),
        "tags_used": used,
        "tags_missing": missing,
        "docs_per_language": dict(lang_docs),
        "num_short_docs_skipped": n_short,
    })


class RedPajamaSubsetDataset(Dataset):
    """Reads the memmap written by build_subset; items match RedPajamaDataset with --pretokenize_dataset."""

    def __init__(self, hparams):
        if hparams.execution_mode != "pretrain":
            raise ValueError("pajama_subset is a pretrain dataset, no other execution modes supported.")
        max_length = hparams.context_length + 1
        self.root = subset_dir(hparams.dataset_dir, hparams.tokenizer, max_length)
        meta_path = os.path.join(self.root, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{meta_path} not found. Build it with: python -m data.nlp.pajama_subset "
                f"--dataset_dir {hparams.dataset_dir} --max_length {max_length} --tokenizer {hparams.tokenizer} --num_docs N"
            )
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["max_length"] != max_length or meta["tokenizer"] != hparams.tokenizer:
            raise ValueError(f"{self.root} was built for max_length={meta['max_length']}, tokenizer={meta['tokenizer']}")
        self.num_docs = meta["num_docs"]
        self.lengths = np.load(os.path.join(self.root, "lengths.npy"))
        rows, width = np.load(os.path.join(self.root, "input_ids.npy"), mmap_mode="r").shape
        if len(self.lengths) != self.num_docs or rows < self.num_docs or width != max_length:
            raise ValueError(f"{self.root} is inconsistent (meta num_docs={self.num_docs}, lengths={len(self.lengths)}, input_ids={rows}x{width}); rebuild it")
        self._input_ids = None  # opened lazily so every DataLoader worker maps the file itself

    def __len__(self):
        return self.num_docs

    def __getitem__(self, idx):
        if self._input_ids is None:
            self._input_ids = np.load(os.path.join(self.root, "input_ids.npy"), mmap_mode="r")
        length = int(self.lengths[idx])
        return {"input_ids": self._input_ids[idx, :length].astype(np.int64).tolist(), "attention_mask": [1] * length}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the pajama_subset dataset used by --dataset_name pajama_subset")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--num_docs", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=257, help="context_length + 1")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--download_workers", type=int, default=16)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)

    script = requests.get(HF_SCRIPT_URL, timeout=60)
    script.raise_for_status()
    languages, snapshots = parse_hf_script_constants(script.text)
    tags = all_tags(languages, snapshots)
    print(f"[pajama_subset] {len(languages)} languages x {len(snapshots)} snapshots -> {len(tags)} tags")

    out_dir = subset_dir(args.dataset_dir, args.tokenizer, args.max_length)
    cache_dir = os.path.join(args.dataset_dir, "pajama_subset", "raw_cache")
    fetch = partial(download_documents, cache_dir=cache_dir, session=requests.Session())
    meta = build_subset(out_dir, args.num_docs, args.max_length, tokenizer, args.tokenizer, tags, fetch, args.seed, args.download_workers)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("tags_used", "tags_missing")}, indent=2))
    print(f"[pajama_subset] wrote {out_dir}; missing shards: {len(meta['tags_missing'])}")


if __name__ == "__main__":
    main()
