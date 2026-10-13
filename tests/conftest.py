import gzip
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, clean_up_tokenization_spaces=False)
    tok.pad_token_id = tok.eos_token_id  # same as RedPajamaDataset / NLP_HF_Collator
    return tok


def write_shard(root, tag, texts):
    """Write a RedPajama-V2 style documents shard (gzip jsonl with raw_content) and return its path."""
    path = os.path.join(root, "documents", f"{tag}.json.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"raw_content": text}) + "\n")
    return path
