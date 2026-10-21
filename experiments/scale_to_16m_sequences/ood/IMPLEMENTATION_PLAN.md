---
title: "OOD evaluation of S2 and T++ at 250k/b64 — implementation plan"
date: 2026-10-21
tags: [ebt, ood, thinking-study, verifier, plan]
status: plan
---

# OOD evaluation of S2 and T++ at 250k/b64 — Implementation Plan

> [!summary] TL;DR
> - Rerun Q2 (self-verification, RQ1) and Q3 Part A (thinking longer, RQ2) for **S2 250k/b64** on the paper's four OOD sets: GSM8K, SQuAD, BigBench Elementary Math QA and BigBench Dyck Languages. **T++ 250k/b64** supplies baseline perplexity and off-trajectory candidates.
> - Score **every token** of `"Question: …\nAnswer: …"` (the repo's `get_ppl` path). Answer-only tokens are a secondary metric from the same run. Use at most 1,024 sequences per set, with the same seeded subset rule as Q2.
> - Reuse the harness and the Q2/Q3 scripts through one new flag, `--data`. Estimated **≈ 1–1.5 GPU-h**, measured in Task 5 before the full run.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the paper's OOD thinking claims hold for our S2 250k/b64 model:
- self-verification (best-of-N by energy) lowers perplexity;
- more inference steps than trained help;
- the gains grow with OOD shift.

The claims come from §4.1.3, Fig. 6a, Fig. 7 and Table 2.

**Architecture:**
- A new `common/ood_data.py` turns the repo's OOD loaders into harness items.
- `harness.eval_dataset()` picks RedPajama val or an OOD set.
- `run_q2.py` and `run_q3.py` gain `--data`; `run_q3.py` also gains `--skip-part-b`.
- Both record an answer-token mask.
- `ood/run_ood.sh` runs the sweep. `ood/collect_ood.py` builds the tables and a Fig. 7-style plot.

**Tech Stack:** PyTorch 2.4, HF `datasets` 3.0.0, `transformers` 4.44.2, numpy, scipy, matplotlib, pytest. Env: `/mnt/data/conda-envs/quangnx/ebt312`.

**Spec:** §0 below, plus `../OOD_EVALUATION_PLAN.md`. Decisions were made in discussion on 2026-10-21. Context: `../report/report.md` (in-distribution results at 250k), `../../thinking_study/q2_verifier_quality/report.md`, `../../thinking_study/q3_energy_landscape/report.md`.

## 0. Spec: decisions and rationale

| Decision | Value | Why |
|---|---|---|
| Models | S2 `s2_b64_16m` step 250000; T++ `tpp_b64_16m` step 250000 | User decision. T++ gives the baseline perplexity (Table 3 analogue) and Q2's off-trajectory candidates |
| OOD sets | `gsm8k` (test), `squad` (squad_v2 validation, answerable only), `bigbench_elementary_math_qa` and `bigbench_dyck_languages` (tasksource/bigbench validation) | The paper's set (§4.1.3, App. D.1.2). Splits are the ones the repo loaders use for evaluation |
| Text format | The repo loaders' `pretrain` format: `Question: {q}\nAnswer: {a}` (SQuAD: `Question: {context}\n{question}\nAnswer: {a}`) | This is what `inference/nlp/generate_text.py:get_ppl` scores |
| Scored tokens | **All** non-pad targets (primary); answer tokens only (secondary) | User decision (a). Answer-only comes free from the same run |
| Tokenization | GPT-NeoX, `truncation=True, max_length=257`, no EOS appended | Same call as the pretraining data (`pajama_subset.tokenize`) |
| Sample | `eval_subset_indices(len(ds), 1024)` (seed 0); all items if the set is smaller | User decision. Same rule as Q2/Q3 |
| RQ1 | Q2 at σ = 3 (training Langevin noise), 32 candidates, default step count | Same protocol as the in-distribution 250k result |
| RQ2 | Q3 Part A at noise 0 and 3, 32 steps, 1,024 sequences; Part B skipped | RQ2 only needs CE after k steps |
| In-distribution point | Existing `results/q2_verifier_quality/s2_b64_16m_step250000_noise3` and `results/q3_energy_landscape/s2_b64_16m_step250000` (Q3 used 256 sequences) | Already computed on the same checkpoint. The sample size differs and is noted in the report |
| OOD shift | ppl(set) / ppl(RedPajama val), both S2 single-sample at σ = 3 | Paper's Fig. 7 definition |
| % improvement | $100\,(1 - e^{\,CE_\text{variant} - CE_\text{default}})$ | Paper reports "% perplexity improvement" (Table 2, Fig. 6a) |

**Pre-registered claims** (paper direction; one seed; thresholds fixed before any OOD number is seen):

| Claim | Supported if | In-distribution reference (250k) |
|---|---|---|
| RQ1 self-verification helps OOD | On ≥ 3 of 4 sets, the best N ∈ {2, 4, 8, 16, 32} has CE(energy pick) < CE(random pick) − 0.02 nats | Best is N = 2: −0.010 → not met |
| RQ2 thinking longer helps OOD | On ≥ 3 of 4 sets, at noise 0 or 3, $\min_{k>3} CE(k) < CE(3) - 0.02$ nats | Best k = 3 at both noises → not met |
| Fig. 7 gains grow with shift | Spearman > 0 between OOD shift and % improvement over the 5 points, for the RQ1 best-N gain and for the RQ2 best-k gain | Descriptive only: 5 points, one seed |

## Global Constraints

- Python: `/mnt/data/conda-envs/quangnx/ebt312/bin/python` only.
- `export HF_HOME=/mnt/data/quangnx/huggingface TMPDIR=/mnt/data/quangnx/tmp`. The root filesystem filled up once (scale-16m crash), so nothing large goes under `/home` or `/tmp`.
- Results go under `experiments/scale_to_16m_sequences/results/ood/`. That is a symlink to `/mnt/data`, and it is git-ignored.
- Work on a new branch `ood-eval` cut from `scale-16m`. Do not push or merge without asking.
- All artifacts are in English. Markdown is Obsidian-flavoured with a 3-bullet TL;DR.
- `--data rp_val` (the default) must reproduce the old behaviour exactly: same dataset object, same indices.
- Always pass `--tpp tpp_b64_16m --tpp-step 250000` explicitly. The `run_q2.py` defaults point at the 105k T++.
- Test suite: `cd /home/quangnx/workspace/ebt/refs/EBT && PYTHONPATH=$PWD:$PWD/experiments/thinking_study/common /mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest tests experiments/thinking_study/common experiments/scale_to_16m_sequences/ood -q`.

## Review Focus

1. **BPE merges across the `Answer:` boundary.** If the prompt's tokens are not a prefix of the full text's tokens, the answer mask is shifted. Pinned in Task 2 (unit test on a Dyck-like string) and in Task 1 (prefix check on every item of every set; expected 0 violations).
2. **Answer truncated away (long SQuAD contexts).** `n_prompt > 257` gives an all-False answer mask for that sequence. The analysis must report `n_tokens = 0` or skip, never crash or emit NaN-as-number. Pinned in Task 2 (truncation test) and Task 4 (zero-answer-token test).
3. **Set smaller than `--n-seqs`.** All items are used, and the config records the real token count. Pinned in Task 3 (`eval_subset_indices` on a short set).
4. **`rp_val` regression.** The default path must return the same `validation_split` object, so old results stay reproducible. Pinned in Task 3.
5. **`analyze_q3.py` on a run without `part_b.npz`.** It must write Part A metrics and omit `part_b`. Pinned in Task 4.

---

## File map

| File | Responsibility | Status |
|---|---|---|
| `experiments/thinking_study/common/ood_data.py` | OOD texts from the repo loaders → `{"input_ids", "n_prompt"}` items | New |
| `experiments/thinking_study/common/harness.py` | `eval_dataset()`, `answer_masks()` | Modify |
| `experiments/thinking_study/common/test_ood_data.py` | Tests for encoding, masks, `eval_dataset` | New |
| `experiments/thinking_study/q2_verifier_quality/run_q2.py` | `--data`; record `answer` | Modify |
| `experiments/thinking_study/q3_energy_landscape/run_q3.py` | `--data`, `--skip-part-b`; record `answer` in Part A | Modify |
| `experiments/thinking_study/q2_verifier_quality/analyze_q2.py` | `answer_only` block | Modify |
| `experiments/thinking_study/q3_energy_landscape/analyze_q3.py` | Optional Part B; `answer_only` block | Modify |
| `experiments/thinking_study/common/test_ood_analyze.py` | Tests for both analyze changes | New |
| `experiments/scale_to_16m_sequences/ood/prepare_data.py` | Download and stats per set → `ood/report/data_stats.json` | New |
| `experiments/scale_to_16m_sequences/ood/run_ood.sh` | The sweep | New |
| `experiments/scale_to_16m_sequences/ood/collect_ood.py` | Tables → `ood/report/result.md`; plot → `ood/report/figures/ood_shift.png` | New |
| `experiments/scale_to_16m_sequences/ood/test_collect_ood.py` | Tests for `pct_improvement` and the row builders | New |
| `experiments/scale_to_16m_sequences/ood/report/report.md` | Findings and verdicts | New |

---

### Task 1: Download the OOD sets and measure them

**Files:**
- Create: `experiments/scale_to_16m_sequences/ood/prepare_data.py`
- Output: `experiments/scale_to_16m_sequences/ood/report/data_stats.json`

**Interfaces:**
- Consumes: `ood_data.ood_texts`, `ood_data.encode`, `ood_data.OOD_SETS` (Task 2). Task 1 is run **after** Task 2's code exists, so execute Task 2 first, then Task 1.
- Produces: the HF cache of all four sets, and `data_stats.json` with, per set: `n_items`, `token_len_quantiles`, `share_truncated`, `share_answer_truncated`, `prefix_violations`.

> [!note] Order
> The numbering follows the spec's logic (data first). Execute in the order **2 → 1 → 3 → 4 → 5 → 6 → 7**.

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Run it (downloads on first call)**

Run:
```bash
cd /home/quangnx/workspace/ebt/refs/EBT/experiments/scale_to_16m_sequences
export HF_HOME=/mnt/data/quangnx/huggingface TMPDIR=/mnt/data/quangnx/tmp
PYTHONPATH=$PWD/../..:$PWD/../thinking_study/common /mnt/data/conda-envs/quangnx/ebt312/bin/python ood/prepare_data.py --out ood/report/data_stats.json
```

Expected:
- All four sets load.
- GSM8K test has 1,319 items (HF card).
- `prefix_violations` is 0 for every set.
- SQuAD has the largest `share_truncated`.

If `tasksource/bigbench` refuses `trust_remote_code` under `datasets` 3.0.0, or a split is missing, stop and ledger a ruling before choosing a source. Do not silently switch to another BigBench mirror. If `prefix_violations` > 0, ledger it and fix `encode` before Task 3.

- [ ] **Step 3: Commit**

```bash
git add experiments/scale_to_16m_sequences/ood/prepare_data.py experiments/scale_to_16m_sequences/ood/report/data_stats.json
git commit -m "feat: OOD set download and length/answer-truncation stats"
```

---

### Task 2: OOD items and answer masks in the harness

**Files:**
- Create: `experiments/thinking_study/common/ood_data.py`
- Modify: `experiments/thinking_study/common/harness.py` (append after `batches`)
- Test: `experiments/thinking_study/common/test_ood_data.py`

**Interfaces:**
- Produces:
  - `ood_data.OOD_SETS: tuple[str, ...]`
  - `ood_data.ANSWER_SEP = "\nAnswer: "`
  - `ood_data.ood_texts(name: str) -> list[str]`
  - `ood_data.encode(text: str, tokenizer) -> dict` with keys `input_ids: list[int]` (≤ 257) and `n_prompt: int` (tokens up to and including `Answer:`)
  - `ood_data.OODDataset(texts, tokenizer)`, a `torch.utils.data.Dataset` of those dicts
  - `harness.eval_dataset(data: str, hparams: dict) -> Dataset`; `"rp_val"` → `validation_split(hparams)`
  - `harness.answer_masks(ds, indices, batch_size)` yields `(B, SEQ_LEN)` bool tensors, aligned with `batches()`. Target j is in the answer iff `j + 1 >= n_prompt`. Items without `n_prompt` are all False.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/thinking_study/common/test_ood_data.py
import pytest
from transformers import AutoTokenizer

import harness
import ood_data


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", clean_up_tokenization_spaces=False)
    t.pad_token_id = t.eos_token_id
    return t


def test_answer_tokens_start_after_prompt(tok):
    item = ood_data.encode("Question: What is 2+3?\nAnswer: 5", tok)
    assert tok.decode(item["input_ids"][item["n_prompt"]:]) == " 5"


def test_prompt_tokens_are_a_prefix_of_the_full_text(tok):
    text = "Question: Complete the sequence: [ ( < >\nAnswer: ) ]"
    item = ood_data.encode(text, tok)
    prompt = tok(text[: text.rindex(ood_data.ANSWER_SEP) + len(ood_data.ANSWER_SEP) - 1])["input_ids"]
    assert item["input_ids"][: item["n_prompt"]] == prompt


def test_long_text_is_truncated_and_answer_falls_outside(tok):
    item = ood_data.encode("Question: " + "word " * 600 + "\nAnswer: 7", tok)
    assert len(item["input_ids"]) == harness.SEQ_LEN + 1
    assert item["n_prompt"] > harness.SEQ_LEN + 1


def test_answer_masks_mark_targets_from_the_first_answer_token():
    ds = [{"input_ids": [1, 2, 3, 4, 5], "n_prompt": 3}, {"input_ids": [1, 2, 3]}]
    m = next(harness.answer_masks(ds, [0, 1], 2))
    assert m.shape == (2, harness.SEQ_LEN)
    assert m[0, :4].tolist() == [False, False, True, True]  # targets 2, 3 predict tokens 3, 4 (the answer)
    assert not m[1].any()


def test_eval_dataset_rp_val_is_the_validation_split(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(harness, "validation_split", lambda hp: sentinel)
    assert harness.eval_dataset("rp_val", {}) is sentinel


def test_eval_dataset_rejects_unknown_names():
    with pytest.raises(ValueError):
        harness.eval_dataset("mmlu", {"tokenizer": "EleutherAI/gpt-neox-20b"})
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd /home/quangnx/workspace/ebt/refs/EBT && PYTHONPATH=$PWD:$PWD/experiments/thinking_study/common /mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest experiments/thinking_study/common/test_ood_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ood_data'`.

- [ ] **Step 3: Implement `ood_data.py`**

```python
"""The paper's OOD eval sets (§4.1.3, Table 3) as harness items.

Texts come from the repo loaders in their `pretrain` format ("Question: ...\nAnswer: ..."), the format
`get_ppl` scores, and are tokenized like the pretraining data (GPT-NeoX, truncated to SEQ_LEN + 1).
`n_prompt` counts the tokens up to and including "Answer:", for the secondary answer-only metric.
"""
from argparse import Namespace

from torch.utils.data import Dataset

from harness import SEQ_LEN

OOD_SETS = ("gsm8k", "squad", "bigbench_elementary_math_qa", "bigbench_dyck_languages")
ANSWER_SEP = "\nAnswer: "


def ood_texts(name: str) -> list[str]:
    from data.nlp.bigbench_dataloader import BigBenchDataset
    from data.nlp.gsm8k_dataloader import GSM8KDataset
    from data.nlp.squad_dataloader import SQuADDataset

    hp = Namespace(dataset_dir="", execution_mode="pretrain")
    make = {"gsm8k": lambda: GSM8KDataset(hp, "test"),
            "squad": lambda: SQuADDataset(hp, "validation"),
            "bigbench_elementary_math_qa": lambda: BigBenchDataset(hp, "validation", "elementary_math_qa"),
            "bigbench_dyck_languages": lambda: BigBenchDataset(hp, "validation", "dyck_languages")}
    ds = make[name]()
    return [ds[i] for i in range(len(ds))]


def encode(text: str, tokenizer) -> dict:
    prompt = text[: text.rindex(ANSWER_SEP) + len(ANSWER_SEP) - 1]  # ends with "Answer:"; the answer keeps its leading space
    return {"input_ids": tokenizer(text, truncation=True, max_length=SEQ_LEN + 1)["input_ids"],
            "n_prompt": len(tokenizer(prompt)["input_ids"])}


class OODDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer):
        self.texts, self.tokenizer = texts, tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return encode(self.texts[i], self.tokenizer)
```

- [ ] **Step 4: Add to `harness.py` after `batches`**

```python
def eval_dataset(data: str, hparams: dict):
    """RedPajama validation ("rp_val") or one of the paper's OOD sets (ood_data.OOD_SETS)."""
    if data == "rp_val":
        return validation_split(hparams)
    from ood_data import OOD_SETS, OODDataset, ood_texts
    if data not in OOD_SETS:
        raise ValueError(f"unknown eval data {data!r}; expected rp_val or one of {OOD_SETS}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hparams["tokenizer"], clean_up_tokenization_spaces=False)
    return OODDataset(ood_texts(data), tok)


def answer_masks(ds, indices: list[int], batch_size: int):
    """Per batch of `batches`, (B, SEQ_LEN) bool: target j is in the answer iff j + 1 >= n_prompt."""
    for start in range(0, len(indices), batch_size):
        items = [ds[i] for i in indices[start : start + batch_size]]
        m = torch.zeros(len(items), SEQ_LEN, dtype=torch.bool)
        for row, item in enumerate(items):
            if "n_prompt" in item:
                m[row, max(item["n_prompt"] - 1, 0):] = True
        yield m
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: same command as Step 2.
Expected: 6 passed. If `test_answer_tokens_start_after_prompt` fails on the decoded string, inspect the tokens, ledger the finding and fix `encode`. Do not change the test's expectation to match.

- [ ] **Step 6: Run the full suite, then commit**

Run: the Global Constraints suite command. Expected: all pass.

```bash
git add experiments/thinking_study/common/ood_data.py experiments/thinking_study/common/harness.py experiments/thinking_study/common/test_ood_data.py
git commit -m "feat: OOD eval sets as harness items with answer-token masks"
```

---

### Task 3: `--data` in run_q2 / run_q3, `--skip-part-b`, answer arrays

**Files:**
- Modify: `experiments/thinking_study/q2_verifier_quality/run_q2.py`
- Modify: `experiments/thinking_study/q3_energy_landscape/run_q3.py`
- Test: `experiments/thinking_study/common/test_ood_data.py` (append)

**Interfaces:**
- Consumes: `harness.eval_dataset`, `harness.answer_masks` (Task 2).
- Produces:
  - Q2 `arrays.npz` gains `answer`: bool, shape `(n_tokens,)`, aligned with `logp[g]`.
  - Q3 `part_a_noise*.npz` gains `answer` with the same alignment.
  - Both `config.json` files gain `"data"`.
  - `run_q3.py --skip-part-b` writes no `part_b.npz`.

- [ ] **Step 1: Write the failing test (short sets use every item)**

```python
def test_subset_of_a_short_set_uses_every_item_once():
    idx = harness.eval_subset_indices(5, 1024)
    assert sorted(idx) == [0, 1, 2, 3, 4]
```

Run: the Task 2 test command. Expected: this test **passes** immediately, because `randperm(n)[:1024]` already handles it. It pins existing behaviour that the OOD path relies on. Record it in the ledger as a characterization test, not a RED step.

- [ ] **Step 2: Edit `run_q2.py`**

- Import `answer_masks, eval_dataset` from `harness` (drop `validation_split`).
- Add the argument:
  ```python
  ap.add_argument("--data", default="rp_val", help="rp_val or an OOD set (ood_data.OOD_SETS)")
  ```
- Replace `val_ds = validation_split(hparams)` with `val_ds = eval_dataset(args.data, hparams)`.
- Add `"answer"` to the `rec` keys.
- Change the loop header to:
  ```python
  for b, ((inputs, targets, mask), amask) in enumerate(zip(batches(val_ds, idx, args.batch_size),
                                                           answer_masks(val_ds, idx, args.batch_size))):
  ```
- Inside the `with torch.no_grad():` block, next to `rec["target"]`, add:
  ```python
  rec["answer"].append(amask.cuda()[mask].cpu())
  ```

- [ ] **Step 3: Edit `run_q3.py`**

- Import `answer_masks, eval_dataset` (drop `validation_split`).
- Add the arguments:
  ```python
  ap.add_argument("--data", default="rp_val", help="rp_val or an OOD set (ood_data.OOD_SETS)")
  ap.add_argument("--skip-part-b", action="store_true", help="Part A only (CE after k steps)")
  ```
- Replace `val_ds = validation_split(hparams)` with `val_ds = eval_dataset(args.data, hparams)`.
- In `part_a`, add `"answer"` to the `rec` keys. Change the loop header to:
  ```python
  for b, ((inputs, targets, mask), amask) in enumerate(zip(batches(val_ds, idx, batch_size),
                                                           answer_masks(val_ds, idx, batch_size))):
  ```
  Next to `rec["logp_final"]`, add:
  ```python
  rec["answer"].append(amask.cuda()[mask].cpu())
  ```
- Wrap the Part B block in `if not args.skip_part_b:`.

`config.json` picks up `data` and `skip_part_b` through `vars(args)`.

- [ ] **Step 4: Smoke both scripts (GPU)**

Run:
```bash
cd /home/quangnx/workspace/ebt/refs/EBT
export HF_HOME=/mnt/data/quangnx/huggingface TMPDIR=/mnt/data/quangnx/tmp PYTHONPATH=$PWD:$PWD/experiments/thinking_study/common CUDA_VISIBLE_DEVICES=<free GPU>
S=/tmp/claude-1016/ood_smoke   # or the session scratchpad
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
$PY experiments/thinking_study/q2_verifier_quality/run_q2.py --model s2_b64_16m --step 250000 --tpp tpp_b64_16m --tpp-step 250000 --noise 3 --data bigbench_dyck_languages --n-seqs 8 --n-candidates 4 --out-dir $S/q2
$PY experiments/thinking_study/q3_energy_landscape/run_q3.py --model s2_b64_16m --step 250000 --data bigbench_dyck_languages --n-seqs-a 8 --n-steps 6 --skip-part-b --out-dir $S/q3
$PY -c "import numpy as np; a=np.load('$S/q2/arrays.npz'); print(a['logp'].shape, a['answer'].shape, a['answer'].mean())"
ls $S/q3
```
Expected:
- `answer` has the same length as `logp.shape[1]`, with mean strictly between 0 and 1.
- `$S/q3` contains `part_a_noise0.npz`, `part_a_noise3.npz` and `config.json`, and **no** `part_b.npz`.

- [ ] **Step 5: rp_val regression check**

Run `run_q2.py` with `--data rp_val --n-seqs 16 --batch-size 16 --n-candidates 2 --model s2_b64_16m --step 250000 --tpp tpp_b64_16m --tpp-step 250000 --noise 3`.

Compare `logp[0]` with the first `n` tokens of the committed 250k run, `results/q2_verifier_quality/s2_b64_16m_step250000_noise3/arrays.npz`. Here `n` is the smoke run's token count.

Expected: equal to `atol=1e-4`. Using `--n-seqs 16` with the default batch size makes batch 0 identical to the full run's batch 0, so the `(B, S, V)` noise draws have the same shape. Seeds depend only on `(seed, b, g)`.

- [ ] **Step 6: Suite, then commit**

```bash
git add experiments/thinking_study/q2_verifier_quality/run_q2.py experiments/thinking_study/q3_energy_landscape/run_q3.py experiments/thinking_study/common/test_ood_data.py
git commit -m "feat: Q2/Q3 runners take --data (rp_val or OOD set) and record answer-token masks; Q3 --skip-part-b"
```

---

### Task 4: Analysis — answer-only metrics, optional Part B

**Files:**
- Modify: `experiments/thinking_study/q2_verifier_quality/analyze_q2.py` (`analyze`)
- Modify: `experiments/thinking_study/q3_energy_landscape/analyze_q3.py` (`analyze`)
- Test: `experiments/thinking_study/common/test_ood_analyze.py`

**Interfaces:**
- Consumes: the `answer` arrays (Task 3).
- Produces:
  - Q2 `metrics.json["answer_only"]` = `{"n_tokens": int, "single_sample_ce": float | None, "best_of_n": dict}`. When `n_tokens == 0`, this is `{"n_tokens": 0}` only.
  - Q3 `metrics.json["part_a_noise{σ}"]["answer_only"]` = `{"n_tokens": int, "ce_after_k_steps": {k: float}}`, with the same zero rule.
  - Q3 `metrics.json` has no `"part_b"` key when `part_b.npz` is absent.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/thinking_study/common/test_ood_analyze.py
import importlib.util
import json
from pathlib import Path

import numpy as np

TS = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_q2(d: Path, answer):
    rng = np.random.default_rng(0)
    G, T = 4, len(answer)
    off = ["tpp_T0.5", "tpp_T1.0", "tpp_T2.0", "onehot_target", "onehot_tpp_argmax", "uniform"]
    np.savez(d / "arrays.npz", e_final=rng.normal(size=(G, T)), e_last_logged=rng.normal(size=(G, T)),
             logp=-rng.random((G, T)) * 5, top1=rng.random((G, T)) > 0.5, entropy=rng.random((G, T)) * 3,
             js=rng.random((3, T)), off_e=rng.normal(size=(6, T)), off_logp=-rng.random((6, T)) * 5,
             target=np.zeros(T, int), tpp_logp=-rng.random(T) * 5, answer=np.array(answer))
    (d / "config.json").write_text(json.dumps({"off_trajectory": off}))


def fake_q3(d: Path, answer, with_part_b=False):
    rng = np.random.default_rng(0)
    S, T = 6, len(answer)
    for noise in (0, 3):
        np.savez(d / f"part_a_noise{noise}.npz", logp=-rng.random((S, T)) * 5, logp_final=-rng.random(T) * 5,
                 energy=rng.normal(size=(S, T)), phi=rng.normal(size=(S, T)), entropy=rng.random((S, T)),
                 gnorm=rng.random((S, T)), tv_next=rng.random((S, T)), cos_prev=rng.random((S, T)),
                 answer=np.array(answer))
    (d / "config.json").write_text(json.dumps({"noises": [0.0, 3.0], "n_train_steps": 3}))


def test_q2_answer_only_scores_only_answer_tokens(tmp_path):
    fake_q2(tmp_path, [False] * 30 + [True] * 10)
    m = load(TS / "q2_verifier_quality/analyze_q2.py").analyze(tmp_path)
    a = np.load(tmp_path / "arrays.npz")
    assert m["answer_only"]["n_tokens"] == 10
    assert np.isclose(m["answer_only"]["single_sample_ce"], -a["logp"][0, 30:].mean())


def test_q2_answer_only_with_no_answer_tokens_is_empty(tmp_path):
    fake_q2(tmp_path, [False] * 40)
    assert load(TS / "q2_verifier_quality/analyze_q2.py").analyze(tmp_path)["answer_only"] == {"n_tokens": 0}


def test_q3_without_part_b_writes_part_a_and_answer_only(tmp_path):
    fake_q3(tmp_path, [False] * 30 + [True] * 10)
    m = load(TS / "q3_energy_landscape/analyze_q3.py").analyze(tmp_path)
    assert "part_b" not in m
    ao = m["part_a_noise0"]["answer_only"]
    assert ao["n_tokens"] == 10 and set(ao["ce_after_k_steps"]) == set(range(1, 7))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd /home/quangnx/workspace/ebt/refs/EBT && PYTHONPATH=$PWD:$PWD/experiments/thinking_study/common /mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest experiments/thinking_study/common/test_ood_analyze.py -q`

Expected:
- 2 Q2 tests FAIL with `KeyError: 'answer_only'`.
- The Q3 test FAILs with `FileNotFoundError` on `part_b.npz`.

- [ ] **Step 3: Implement in `analyze_q2.py`, at the end of `analyze`, before `return m`**

```python
    if "answer" in a.files:  # OOD runs: secondary metric on answer tokens only
        ans = a["answer"].astype(bool)
        m["answer_only"] = {"n_tokens": int(ans.sum())}
        if ans.any():
            m["answer_only"] |= {"single_sample_ce": float(ce[0, ans].mean()),
                                 "best_of_n": best_of_n(a["e_final"][:, ans], ce[:, ans])}
```

- [ ] **Step 4: Implement in `analyze_q3.py`, `analyze`**

```python
def analyze(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    m = {"config": cfg}
    for noise in cfg["noises"]:
        a = np.load(run_dir / f"part_a_noise{noise:g}.npz")
        m[f"part_a_noise{noise:g}"] = part_a_metrics(a, cfg["n_train_steps"])
        if "answer" in a.files:  # OOD runs: CE after k steps on answer tokens only
            ans = a["answer"].astype(bool)
            ao = {"n_tokens": int(ans.sum())}
            if ans.any():
                ce_after = -np.concatenate([a["logp"][1:, ans], a["logp_final"][None, ans]], 0)
                ao["ce_after_k_steps"] = {k: float(ce_after[k - 1].mean()) for k in range(1, ce_after.shape[0] + 1)}
            m[f"part_a_noise{noise:g}"]["answer_only"] = ao
    if (run_dir / "part_b.npz").exists():
        m["part_b"] = part_b_metrics(np.load(run_dir / "part_b.npz"))
    return m
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: the Step 2 command. Expected: 3 passed.

Then rerun both analyzers on the committed 250k in-distribution runs. Diff each new `metrics.json` against the old one, ignoring `config.seconds`. Expected: identical, because those arrays have no `answer` key.

- [ ] **Step 6: Suite, then commit**

```bash
git add experiments/thinking_study/q2_verifier_quality/analyze_q2.py experiments/thinking_study/q3_energy_landscape/analyze_q3.py experiments/thinking_study/common/test_ood_analyze.py
git commit -m "feat: answer-only Q2/Q3 metrics for OOD runs; analyze_q3 works without Part B"
```

---

### Task 5: Sweep script, smoke run, timing

**Files:**
- Create: `experiments/scale_to_16m_sequences/ood/run_ood.sh`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: `results/ood/q2_verifier_quality/s2_b64_16m_step250000_{data}_noise3/` and `results/ood/q3_energy_landscape/s2_b64_16m_step250000_{data}/`, each with `metrics.json`.

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# OOD eval of S2 250k/b64: Q2 at σ=3 (RQ1, self-verification) and Q3 Part A (RQ2, thinking longer) per OOD set.
# Env: GPU_ID (required); N_SEQS (default 1024); DATASETS (default: all four); OUT (default results/ood).
set -euo pipefail
OOD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$OOD/.." && pwd)"
TS="$(cd "$WS/../thinking_study" && pwd)"
REPO="$(cd "$WS/../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export TMPDIR=/mnt/data/quangnx/tmp HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
N_SEQS="${N_SEQS:-1024}"
OUT="${OUT:-$WS/results/ood}"
DATASETS="${DATASETS:-gsm8k squad bigbench_elementary_math_qa bigbench_dyck_languages}"
cd "$REPO"
for data in $DATASETS; do
  echo "=== $data $(date -Is)"
  q2="$OUT/q2_verifier_quality/s2_b64_16m_step250000_${data}_noise3"
  $PY "$TS/q2_verifier_quality/run_q2.py" --model s2_b64_16m --step 250000 --tpp tpp_b64_16m --tpp-step 250000 \
    --noise 3 --data "$data" --n-seqs "$N_SEQS" --out-dir "$q2"
  $PY "$TS/q2_verifier_quality/analyze_q2.py" "$q2" > "$q2/metrics.log"
  q3="$OUT/q3_energy_landscape/s2_b64_16m_step250000_${data}"
  $PY "$TS/q3_energy_landscape/run_q3.py" --model s2_b64_16m --step 250000 --data "$data" --n-seqs-a "$N_SEQS" \
    --skip-part-b --out-dir "$q3"
  $PY "$TS/q3_energy_landscape/analyze_q3.py" "$q3"
done
echo "=== done $(date -Is)"
```

- [ ] **Step 2: Smoke run with timing**

Run:
```bash
time GPU_ID=<free GPU> N_SEQS=16 OUT=<scratchpad>/ood_smoke bash experiments/scale_to_16m_sequences/ood/run_ood.sh
```

Expected:
- Four sets finish, and each output dir has a `metrics.json`.
- Extrapolate the full run as (smoke seconds per set) × 64 (1,024 / 16), summed over the sets.
- Ledger the estimate. If it exceeds 3 GPU-h, stop and ask before Task 6.

- [ ] **Step 3: Commit**

```bash
git add experiments/scale_to_16m_sequences/ood/run_ood.sh
git commit -m "feat: OOD sweep script for S2 250k/b64 (Q2 σ=3 and Q3 Part A per set)"
```

---

### Task 6: Full run and collection

**Files:**
- Create: `experiments/scale_to_16m_sequences/ood/collect_ood.py`
- Test: `experiments/scale_to_16m_sequences/ood/test_collect_ood.py`
- Output: `ood/report/result.md`, `ood/report/figures/ood_shift.png`

**Interfaces:**
- Consumes: Task 5 outputs, plus the in-distribution 250k runs listed in §0.
- Produces: `pct_improvement(ce_default, ce_variant) -> float`, `rq1_row(q2_metrics) -> dict`, `rq2_row(q3_metrics, noise) -> dict`, and the two output files.

- [ ] **Step 1: Start the full run in the background**

Run:
```bash
GPU_ID=<free GPU> bash experiments/scale_to_16m_sequences/ood/run_ood.sh > experiments/scale_to_16m_sequences/runs/ood_run.log 2>&1
```
Write tasks 6.2–6.4 while it runs.

- [ ] **Step 2: Write the failing tests**

```python
# experiments/scale_to_16m_sequences/ood/test_collect_ood.py
import math

import collect_ood


def test_pct_improvement_matches_perplexity_ratio():
    assert collect_ood.pct_improvement(2.0, 2.0) == 0.0
    assert math.isclose(collect_ood.pct_improvement(2.0, 2.0 - math.log(2)), 50.0)
    assert collect_ood.pct_improvement(2.0, 2.1) < 0


def test_rq1_row_picks_the_best_n():
    bon = {str(n): {"ce_energy_pick": 3.0 + d, "ce_random": 3.0} for n, d in [(2, -0.03), (4, -0.01), (8, 0.02)]}
    row = collect_ood.rq1_row({"e_final": {"best_of_n": bon}})
    assert row["best_n"] == 2 and math.isclose(row["best_delta"], -0.03)


def test_rq2_row_uses_trained_k_as_default_and_best_longer_k():
    ce = {"1": 6.0, "2": 4.0, "3": 3.5, "4": 3.6, "5": 3.45, "6": 3.55}
    row = collect_ood.rq2_row({"part_a_noise0": {"ce_after_k_steps": ce, "best_k": {"k": 5}}}, 0)
    assert row["ce_k3"] == 3.5 and row["best_longer_k"] == 5 and math.isclose(row["best_longer_delta"], -0.05)
```

Run: `cd experiments/scale_to_16m_sequences/ood && PYTHONPATH=. /mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest test_collect_ood.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect_ood'`.

- [ ] **Step 3: Implement `collect_ood.py`**

```python
"""OOD tables (report/result.md) and the Fig. 7-style plot (report/figures/ood_shift.png).

Usage: python ood/collect_ood.py
"""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OOD = Path(__file__).resolve().parent
WS = OOD.parent
TRAINED_K = 3
SETS = ["rp_val", "gsm8k", "squad", "bigbench_elementary_math_qa", "bigbench_dyck_languages"]
Q2 = {s: WS / "results/ood/q2_verifier_quality" / f"s2_b64_16m_step250000_{s}_noise3" for s in SETS[1:]}
Q2["rp_val"] = WS / "results/q2_verifier_quality/s2_b64_16m_step250000_noise3"
Q3 = {s: WS / "results/ood/q3_energy_landscape" / f"s2_b64_16m_step250000_{s}" for s in SETS[1:]}
Q3["rp_val"] = WS / "results/q3_energy_landscape/s2_b64_16m_step250000"


def pct_improvement(ce_default: float, ce_variant: float) -> float:
    """Paper's % perplexity improvement: 100 (1 - ppl_variant / ppl_default)."""
    return 100 * (1 - math.exp(ce_variant - ce_default))


def rq1_row(q2: dict) -> dict:
    bon = q2["e_final"]["best_of_n"]
    deltas = {int(n): v["ce_energy_pick"] - v["ce_random"] for n, v in bon.items()}
    n = min(deltas, key=deltas.get)
    return {"deltas": deltas, "best_n": n, "best_delta": deltas[n],
            "best_pct": pct_improvement(bon[str(n)]["ce_random"], bon[str(n)]["ce_energy_pick"])}


def rq2_row(q3: dict, noise: int) -> dict:
    ce = {int(k): v for k, v in q3[f"part_a_noise{noise}"]["ce_after_k_steps"].items()}
    longer = {k: v for k, v in ce.items() if k > TRAINED_K}
    k = min(longer, key=longer.get)
    return {"ce": ce, "ce_k3": ce[TRAINED_K], "best_k": q3[f"part_a_noise{noise}"]["best_k"]["k"],
            "best_longer_k": k, "best_longer_delta": longer[k] - ce[TRAINED_K],
            "best_longer_pct": pct_improvement(ce[TRAINED_K], longer[k])}


def load(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text())


def main():
    q2 = {s: load(Q2[s]) for s in SETS}
    q3 = {s: load(Q3[s]) for s in SETS}
    ppl = {s: math.exp(q2[s]["single_sample"]["ce"]) for s in SETS}
    shift = {s: ppl[s] / ppl["rp_val"] for s in SETS}
    r1 = {s: rq1_row(q2[s]) for s in SETS}
    r2 = {(s, n): rq2_row(q3[s], n) for s in SETS for n in (0, 3)}
    L = ["# OOD results — S2 250k/b64 (generated by collect_ood.py)", "",
         "| Set | Tokens | T++ ppl | S2 ppl (σ=3, 1 sample) | OOD shift |", "|---|---|---|---|---|"]
    for s in SETS:
        L.append(f"| {s} | {q2[s]['config']['n_tokens']} | {math.exp(q2[s]['single_sample']['tpp_ce']):.2f} | "
                 f"{ppl[s]:.2f} | {shift[s]:.2f} |")
    L += ["", "RQ1 — CE(energy pick) − CE(random pick), σ = 3:", "",
          "| Set | N=2 | 4 | 8 | 16 | 32 | best N | best % ppl improvement |", "|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        d = r1[s]["deltas"]
        L.append(f"| {s} | " + " | ".join(f"{d[n]:+.3f}" for n in (2, 4, 8, 16, 32)) +
                 f" | {r1[s]['best_n']} | {r1[s]['best_pct']:+.1f}% |")
    L += ["", "RQ2 — CE after k steps (trained k = 3):", "",
          "| Set | noise | k=3 | 4 | 5 | 6 | 8 | 16 | 32 | best k | best k>3 Δ | % |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        for n in (0, 3):
            r = r2[(s, n)]
            L.append(f"| {s} | {n} | " + " | ".join(f"{r['ce'][k]:.3f}" for k in (3, 4, 5, 6, 8, 16, 32)) +
                     f" | {r['best_k']} | {r['best_longer_delta']:+.3f} | {r['best_longer_pct']:+.1f}% |")
    L += ["", "Answer tokens only (secondary):", "",
          "| Set | answer tokens | S2 CE | BoN-32 Δ | Q3 noise 0 CE k=3 → best k>3 |", "|---|---|---|---|---|"]
    for s in SETS[1:]:
        ao = q2[s].get("answer_only", {"n_tokens": 0})
        a3 = q3[s]["part_a_noise0"].get("answer_only", {"n_tokens": 0})
        if ao["n_tokens"] == 0:
            L.append(f"| {s} | 0 | – | – | – |")
            continue
        b = ao["best_of_n"]["32"]
        ce3 = {int(k): v for k, v in a3["ce_after_k_steps"].items()}
        L.append(f"| {s} | {ao['n_tokens']} | {ao['single_sample_ce']:.3f} | {b['ce_energy_pick'] - b['ce_random']:+.3f} | "
                 f"{ce3[TRAINED_K]:.3f} → {min(v for k, v in ce3.items() if k > TRAINED_K):.3f} |")
    x = [shift[s] for s in SETS]
    for label, y in [("RQ1 best-N", [r1[s]["best_pct"] for s in SETS]),
                     ("RQ2 best k>3, noise 3", [r2[(s, 3)]["best_longer_pct"] for s in SETS])]:
        L.append(f"\nSpearman(OOD shift, {label} % improvement) = {spearmanr(x, y).statistic:+.2f} (5 points)")
    (OOD / "report").mkdir(exist_ok=True)
    (OOD / "report/result.md").write_text("\n".join(L) + "\n")
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(x, [r1[s]["best_pct"] for s in SETS], "o", label="self-verification (best N, σ=3)")
    ax.plot(x, [r2[(s, 3)]["best_longer_pct"] for s in SETS], "s", label="thinking longer (best k>3, noise 3)")
    for s, xi in zip(SETS, x):
        ax.annotate(s.replace("bigbench_", "BB "), (xi, r1[s]["best_pct"]), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("OOD shift = ppl(set) / ppl(RedPajama val)")
    ax.set_ylabel("% perplexity improvement")
    ax.legend(fontsize=7)
    (OOD / "report/figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OOD / "report/figures/ood_shift.png", dpi=160)
    print("\n".join(L))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: the Step 2 command. Expected: 3 passed. Then run the full suite.

- [ ] **Step 5: After the run finishes, collect**

Run:
```bash
cd /home/quangnx/workspace/ebt/refs/EBT/experiments/scale_to_16m_sequences && /mnt/data/conda-envs/quangnx/ebt312/bin/python ood/collect_ood.py
```

Expected:
- `report/result.md` with 5 sets in every table.
- The `rp_val` RQ1 row reproduces the in-distribution deltas in `../report/report.md` §2.2 (−0.010 at N = 2, +0.099 at N = 32).

- [ ] **Step 6: Commit**

```bash
git add experiments/scale_to_16m_sequences/ood/collect_ood.py experiments/scale_to_16m_sequences/ood/test_collect_ood.py
git add -f experiments/scale_to_16m_sequences/ood/report/result.md experiments/scale_to_16m_sequences/ood/report/figures/ood_shift.png
git commit -m "feat: OOD result tables and shift plot for S2 250k/b64"
```

---

### Task 7: Report

**Files:**
- Create: `experiments/scale_to_16m_sequences/ood/report/report.md`

- [ ] **Step 1: Write the report**

Use the structure of `../report/report.md`:
- TL;DR (3 bullets).
- §1 Setup: the §0 table, plus `data_stats.json` highlights (size, truncation, answer-truncated share).
- §2 Results: ppl and shift; RQ1; RQ2; answer-only; Fig. 7 plot.
- §3 Verdicts against the pre-registered claims, using the thresholds as written in §0.
- §4 Interpretation.
- Limitations:
  - one seed;
  - 8× less data and a smaller batch than the paper's thinking models;
  - Q3 rp_val used 256 sequences vs 1,024 for OOD;
  - all-token scoring includes question tokens;
  - SQuAD truncation;
  - paper's exact scored tokens unstated (all-token scoring is our reading of `get_ppl`);
  - "max thinking" (more steps + BoN combined) not measured.
- §5 Reproduction commands.

Every number must come from `result.md`, `data_stats.json` or a `metrics.json`. Do not round thresholds in the verdict direction.

- [ ] **Step 2: Commit**

```bash
git add -f experiments/scale_to_16m_sequences/ood/report/report.md
git commit -m "docs: OOD evaluation report for S2 250k/b64 (RQ1 self-verification, RQ2 thinking longer)"
```
