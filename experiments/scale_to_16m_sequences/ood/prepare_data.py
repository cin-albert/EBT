"""Download the paper's OOD sets and record size, length and answer-truncation stats.

Usage: python ood/prepare_data.py --out ood/report/data_stats.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from harness import SEQ_LEN
from ood_data import ANSWER_SEP, OOD_SETS, encode, ood_texts


def stats(name: str, tok) -> dict:
    texts = ood_texts(name)
    items = [encode(t, tok) for t in texts]
    full = [len(tok(t)["input_ids"]) for t in texts]
    violations = sum(tok(t)["input_ids"][: it["n_prompt"]] != tok(t[: t.rindex(ANSWER_SEP) + len(ANSWER_SEP) - 1])["input_ids"]
                     for t, it in zip(texts, items))
    return {"n_items": len(texts),
            "token_len_quantiles": {q: float(np.quantile(full, q)) for q in (0.1, 0.5, 0.9, 1.0)},
            "share_truncated": float(np.mean([n > SEQ_LEN + 1 for n in full])),
            "share_answer_truncated": float(np.mean([it["n_prompt"] >= SEQ_LEN + 1 for it in items])),
            "prefix_violations": int(violations)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    out = {name: stats(name, tok) for name in OOD_SETS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
