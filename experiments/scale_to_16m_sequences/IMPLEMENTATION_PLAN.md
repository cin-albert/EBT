---
title: "Scale S2 vs Transformer++ to 16M sequences — implementation plan"
date: 2026-10-16
tags: [ebt, scaling, system2, plan]
status: plan
---

# Scale S2 vs Transformer++ to 16M sequences — Implementation Plan

> [!summary] TL;DR
> - Train EBT-S2 and Transformer++ (xxs) with **DDP on 2 GPUs, global batch 64, LR 0.0012, 250k steps = 16M sequences** (4.8× the existing 105k/batch-32 runs, which serve as the 3.36M-sequence point).
> - All runs validate on **the same 33,320 documents** as the reproduction runs. The dataset is rebuilt to ~16.1M docs as an exact superset (same shard order), and the old validation indices are excluded from training.
> - Measure the paper's three directional claims (S2–T++ gap shrinks; verifier improves; thinking helps more) with the existing Q2/Q3 tools. Estimated **≈ 32–35 h wall-clock** on 2 GPUs, plus 3–4 h of CPU-only data build.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the paper's thinking-scaling direction (App. C.2, Fig. B.1, Table 3) shows up at xxs when the S2 training data grows from 3.36M to 16M sequences.

**Architecture:**
- Reuse the repo trainer unchanged except for one opt-in flag, `--fixed_val_indices`, which pins the validation set.
- Reuse `data.nlp.pajama_subset` to build a larger superset dataset.
- Reuse the thinking-study harness and the Q2/Q3 scripts, generalized to accept the new runs.
- New code lives in `experiments/scale_to_16m_sequences/`. Heavy data and checkpoints go under `/mnt/data`.

**Tech Stack:** PyTorch 2.4 + Lightning (DDP), the env `/mnt/data/conda-envs/quangnx/ebt312`, numpy, pytest.

**Spec:** §0 below. It records decisions made in discussion on 2026-10-16. Context: `../thinking_study/THINKING_STUDY_PLAN.md`, `../thinking_study/q2_verifier_quality/report.md`, `../thinking_study/q3_energy_landscape/report.md`, `../reproduce_xxs_scaling/report/report.md` §8.

## 0. Spec: decisions and rationale

| Decision | Value | Why |
|---|---|---|
| Models | Transformer++ (T++) and EBT-S2 only | The question is S2 thinking vs T++; S1 is not needed |
| Parallelism | DDP, 2 GPUs, 32 per device → global batch 64 | Chosen by the user; ≈ 2× faster per sequence than 1 GPU |
| LR | 0.0012 (unchanged) | Paper keeps 0.0012 at batch 32 (App. D.1.1) and 128 (Table D.4) |
| Large point | 250k steps × 64 = **16.0M sequences** | The user's data budget (originally 500k × 32) |
| Small point | The existing 105k × 32 runs (3.36M sequences) and their Q2/Q3 results | Decided by the user: batch 32 → 64 is not treated as a confound (both models share it; see Limitations) |
| All other hparams | Identical to `../reproduce_xxs_scaling/run_baseline_xxs.sh` and `run_ebt_s2_xxs.sh` (paper Table D.4 S2 MCMC settings) | Comparability with the reproduction |
| Warmup | 10k steps (4% of the 250k run; 9.5% of the 105k runs) | Same as the paper and the reproduction |
| Validation | The reproduction's 33,320 documents, fixed by index | One common yardstick for the 105k/b32 and 16M/b64 runs |
| Data | ~16.1M documents, the same shard order (seed 33) as the 3.4M build | No repetition (under one epoch); the old data is an exact prefix |
| Checkpoints | Every validation, every 25k steps (10 per run) | Curves over training; Q2 at intermediate checkpoints |
| Seeds | One seed per run | Seed variance stays unmeasured; the gap threshold below is fixed instead |

### Pre-registered expectations (the paper's direction)

The S2 – T++ validation CE gap is +0.247 nats at 3.36M sequences (batch 32). The paper reports +0.064 at 128M sequences (1M × 128; Table 3: ppl 33.43 vs 31.36).

| Claim | Supported if |
|---|---|
| Gap shrinks with data (App. C.2) | gap(16M, b64) < gap(3.36M, b32) = 0.247 by more than 0.03 nats |
| Verifier improves with training (Fig. B.1) | Q2 BoN-32 verifier efficiency and pairwise consistency at σ = 3 increase from the 105k/b32 checkpoint (−0.20; 0.623) to 16M, and along the 16M run's checkpoints |
| Thinking helps more | Q3: the CE penalty of extra steps (k = 32 vs trained k) shrinks, or the best k grows |

A log-linear interpolation between 0.247 and 0.064 predicts ≈ 0.17 nats at 16M. This is a rough guide only, because the paper's setting differs (full corpus, batch 128, and possibly model "Small"; see the reproduction report §8.3).

## Global Constraints

- Python: `/mnt/data/conda-envs/quangnx/ebt312/bin/python`; `HF_HOME=/mnt/data/quangnx/huggingface`.
- At most **2 GPUs**, as a same-NUMA pair (`4,5` or `6,7`; `nvidia-smi topo -m` shows no NVLink, and these pairs are `NODE`). Both must be idle (< 1000 MiB used) at launch.
- Per device batch 32, `--accumulate_grad_batches 1`, `--peak_learning_rate 0.0012`, `--set_matmul_precision medium`, no `--lr_scaling_rule`.
- Validation is always the 33,320 indices in `experiments/scale_to_16m_sequences/val_indices_repro_33320.npy` (committed; not under the `data/` symlink).
- Heavy artifacts (raw shards, memmap, checkpoints) go under `/mnt/data/quangnx/research/ebt/scale_16m/` through the symlinks `data/` and `runs/`. Do not commit them.
- Branch `scale-16m`, created from `thinking-study`. Commit locally per task; **never push**.
- Every file written is in English. Markdown is Obsidian-friendly with a 3-bullet TL;DR. Python follows the `python-guidelines-research` skill (short, typed where useful, no over-engineering).
- Do not modify `../reproduce_xxs_scaling/` (its data and runs are read-only inputs).

## Review Focus

1. **The rebuilt dataset is not an exact superset** (a shard that was downloadable before now returns 404, or a transient error reorders consumption). The old validation indices would then point at different documents. `verify_dataset.py` (Task 4) must fail loudly on any row mismatch.
2. **Epoch rollover.** If the training rows are fewer than 250k × 64, Lightning starts epoch 1, which repeats data and changes checkpoint names (`epoch=epoch=1-...`). Task 4 asserts the training rows are ≥ 16,000,000, and Task 3's `ckpt_path` globs any epoch.
3. **RAM blow-up from Python-list indices** copied into 2 × 12 DataLoader workers (16M Python ints ≈ 0.5 GB per process, which copy-on-write turns into real copies). `fixed_val_split` must return numpy index arrays; Task 2 tests this.
4. **Old checkpoints must behave exactly as before.** Their hparams have no `fixed_val_indices`, so `validation_split` must fall back to the seeded `random_split`. Task 3 re-runs `../thinking_study/common/gate.py`, which must still pass.
5. **The eval subset must be the same documents across runs.** Q2/Q3 index *positions* within the validation set (`eval_subset_indices`), so the fixed split must keep the old order. Task 4 checks that position i of the new validation set is the same document as position i of the old one.

---

## File structure

| Path | Responsibility | New / modified |
|---|---|---|
| `data/fixed_val_split.py` (repo root) | `fixed_val_split(full_ds, path) -> (train Subset, val Subset)` | New |
| `base_model_trainer.py` (`setup`, ~l. 470–476) | Use `fixed_val_split` when `hparams.fixed_val_indices` is set | Modified |
| `train_model.py` (argparse) | `--fixed_val_indices` flag | Modified |
| `tests/test_fixed_val_split.py` | Unit tests for the split | New |
| `experiments/thinking_study/common/harness.py` | Run registry for new runs, any-epoch checkpoint glob, fixed-val support | Modified |
| `experiments/thinking_study/common/test_harness.py` | Unit tests for the registry and the split fallback | New |
| `experiments/thinking_study/q2_verifier_quality/run_q2.py`, `q3_energy_landscape/run_q3.py` | Accept the new run keys; Q2 takes a matched T++ | Modified |
| `experiments/scale_to_16m_sequences/.gitignore` | Ignore `data`, `runs`, `*.npz`, `__pycache__` | New |
| `experiments/scale_to_16m_sequences/export_val_indices.py` | Write `data/val_indices_repro_33320.npy` from the reproduction split | New |
| `experiments/scale_to_16m_sequences/verify_dataset.py` | Superset, order, disjointness and size checks | New |
| `experiments/scale_to_16m_sequences/common.sh`, `run_tpp.sh`, `run_s2.sh` | DDP launch settings | New |
| `experiments/scale_to_16m_sequences/benchmark_ddp.py` | 50/350-step timing → ETA | New |
| `experiments/scale_to_16m_sequences/run_all.sh` | Sequential training queue (T++ then S2) | New |
| `experiments/scale_to_16m_sequences/run_eval.sh` | Q2/Q3 sweep for the new checkpoints | New |
| `experiments/scale_to_16m_sequences/collect.py` | Curves and gap tables → `report/result.md` | New |
| `experiments/scale_to_16m_sequences/report/report.md` | Findings | New |

The validation-index file (33,320 int64 values, 267 KB) is committed in the workspace root, not under `data/`, so it survives without the symlink.

---

### Task 1: Branch and workspace scaffold

**Files:**
- Create: `experiments/scale_to_16m_sequences/.gitignore`
- Create: symlinks `experiments/scale_to_16m_sequences/data`, `experiments/scale_to_16m_sequences/runs`

**Interfaces:**
- Produces: `WS=experiments/scale_to_16m_sequences`; `$WS/data` → `/mnt/data/quangnx/research/ebt/scale_16m/data`; `$WS/runs` → `/mnt/data/quangnx/research/ebt/scale_16m/runs`.

- [ ] **Step 1: Create the branch from `thinking-study`**

```bash
cd /home/quangnx/workspace/ebt/refs/EBT
git checkout thinking-study && git checkout -b scale-16m
```
Expected: `Switched to a new branch 'scale-16m'`.

- [ ] **Step 2: Create the storage and symlinks**

```bash
WS=experiments/scale_to_16m_sequences
mkdir -p /mnt/data/quangnx/research/ebt/scale_16m/{data,runs}
ln -s /mnt/data/quangnx/research/ebt/scale_16m/data $WS/data
ln -s /mnt/data/quangnx/research/ebt/scale_16m/runs $WS/runs
printf 'data\nruns\nlogs/*.log\n*.npz\n__pycache__/\n' > $WS/.gitignore
```

- [ ] **Step 3: Commit the plan and the scaffold**

```bash
git add $WS/.gitignore $WS/IMPLEMENTATION_PLAN.md
git commit -m "chore: scale_to_16m_sequences workspace and implementation plan

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: `--fixed_val_indices` in the trainer

**Files:**
- Create: `data/fixed_val_split.py`
- Modify: `base_model_trainer.py` (inside `setup`, the branch that calls `random_split` for `pajama_subset`)
- Modify: `train_model.py` (argparse, next to `--validation_split_pct`)
- Test: `tests/test_fixed_val_split.py`

**Interfaces:**
- Produces: `fixed_val_split(full_ds: Dataset, val_indices_path: str) -> tuple[Subset, Subset]`. Both `Subset.indices` are `np.ndarray[int64]`. The validation order equals the file order. It raises `ValueError` on duplicates, out-of-range values or non-1-D input.
- Produces: CLI flag `--fixed_val_indices PATH` (default `""`). It is stored in the checkpoint hparams as `fixed_val_indices`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fixed_val_split.py
import numpy as np
import pytest

from data.fixed_val_split import fixed_val_split


def _save(tmp_path, idx):
    path = tmp_path / "val.npy"
    np.save(path, np.asarray(idx, dtype=np.int64))
    return str(path)


def test_split_is_disjoint_complete_and_keeps_val_order(tmp_path):
    full = list(range(10))
    train, val = fixed_val_split(full, _save(tmp_path, [7, 2, 5]))
    assert [val[i] for i in range(len(val))] == [7, 2, 5]
    assert sorted(train[i] for i in range(len(train))) == [0, 1, 3, 4, 6, 8, 9]


def test_indices_are_numpy_not_python_lists(tmp_path):
    train, val = fixed_val_split(list(range(10)), _save(tmp_path, [1, 3]))
    assert isinstance(train.indices, np.ndarray) and isinstance(val.indices, np.ndarray)


@pytest.mark.parametrize("bad", [[1, 1], [10], [-1], [[1, 2]]])
def test_rejects_bad_indices(tmp_path, bad):
    with pytest.raises(ValueError):
        fixed_val_split(list(range(10)), _save(tmp_path, bad))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd /home/quangnx/workspace/ebt/refs/EBT && /mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest tests/test_fixed_val_split.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.fixed_val_split'`.

- [ ] **Step 3: Implement**

```python
# data/fixed_val_split.py
"""Train/validation split with validation indices fixed by a file (keeps a validation set across dataset rebuilds)."""
import numpy as np
from torch.utils.data import Dataset, Subset


def fixed_val_split(full_ds: Dataset, val_indices_path: str) -> tuple[Subset, Subset]:
    """Return (train, val); val keeps the file order, train is every other index in ascending order.

    Indices stay numpy arrays so forked DataLoader workers share them instead of copying Python ints.
    """
    val_idx = np.load(val_indices_path)
    n = len(full_ds)
    if val_idx.ndim != 1 or len(np.unique(val_idx)) != len(val_idx) or val_idx.min() < 0 or val_idx.max() >= n:
        raise ValueError(f"{val_indices_path}: need unique 1-D indices in [0, {n}), got shape {val_idx.shape}")
    is_val = np.zeros(n, dtype=bool)
    is_val[val_idx] = True
    return Subset(full_ds, np.flatnonzero(~is_val)), Subset(full_ds, val_idx.astype(np.int64))
```

In `base_model_trainer.py`, replace the `random_split` lines inside the `if self.hparams.dataset_name in [...]` branch of `setup`:

```python
                self.create_full_ds()
                if self.hparams.get("fixed_val_indices"):  # absent in older checkpoints' hparams
                    self.train_ds, self.val_ds = fixed_val_split(self.full_ds, self.hparams.fixed_val_indices)
                else:
                    train_samples = int(len(self.full_ds) * (1 - self.hparams.validation_split_pct))
                    valid_samples = len(self.full_ds) - train_samples
                    self.train_ds, self.val_ds = random_split(self.full_ds, [train_samples, valid_samples])
```

Add `from data.fixed_val_split import fixed_val_split` next to the other `data.*` imports at the top of `base_model_trainer.py`. In `train_model.py`, after the `--validation_split_pct` argument:

```python
    parser.add_argument("--fixed_val_indices", help="path to a .npy of validation indices into the full dataset; overrides validation_split_pct", type=str, default="")
```

- [ ] **Step 4: Run the tests and the existing suite**

Run: `/mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest tests -q`
Expected: all pass, including the 6 new tests (3 + 3 parametrized).

- [ ] **Step 5: Commit**

```bash
git add data/fixed_val_split.py base_model_trainer.py train_model.py tests/test_fixed_val_split.py
git commit -m "feat: --fixed_val_indices pins the validation set across dataset rebuilds

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: Harness and Q2/Q3 scripts accept the new runs

**Files:**
- Modify: `experiments/thinking_study/common/harness.py` (`CKPT_ROOT`, `RUN_DIRS`, `ckpt_path`, `validation_split`)
- Modify: `experiments/thinking_study/q2_verifier_quality/run_q2.py` (l. 41–53)
- Modify: `experiments/thinking_study/q3_energy_landscape/run_q3.py` (l. 177)
- Test: `experiments/thinking_study/common/test_harness.py`

**Interfaces:**
- Consumes: `fixed_val_split` (Task 2).
- Produces: `RUN_DIRS: dict[str, tuple[Path, str]]`, mapping key → (checkpoint root, directory glob). New keys: `tpp_b64_16m`, `s2_b64_16m`. These match `--run_name` values `tpp-b64-16m`, `s2-b64-16m` (Task 6).
- Produces: `ckpt_path(model: str, step: int) -> Path` (any epoch); `validation_split(hparams: dict) -> Subset` (fixed indices when present).
- Produces: `run_q2.py --model KEY --tpp KEY --tpp-step N`; `run_q3.py --model KEY`.

- [ ] **Step 1: Write the failing tests**

```python
# experiments/thinking_study/common/test_harness.py
import numpy as np
import pytest

import harness


def test_ckpt_path_finds_any_epoch_and_new_run_keys(tmp_path, monkeypatch):
    run = tmp_path / "s2-b64-16m_2026-10-17_00-00-00_"
    run.mkdir()
    (run / "epoch=epoch=1-step=step=250000-valid_loss=valid_loss=3.9000.ckpt").touch()
    monkeypatch.setitem(harness.RUN_DIRS, "s2_b64_16m", (tmp_path, "s2-b64-16m_20*"))
    assert harness.ckpt_path("s2_b64_16m", 250000).name.startswith("epoch=epoch=1-step=step=250000")


def test_run_glob_does_not_match_longer_run_name(tmp_path, monkeypatch):
    for name in ("s2-b64-16m_2026-10-17_00-00-00_", "s2-b64-16m-seed2_2026-10-17_00-00-00_"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "epoch=epoch=0-step=step=25000-valid_loss=valid_loss=4.0.ckpt").touch()
    monkeypatch.setitem(harness.RUN_DIRS, "s2_b64_16m", (tmp_path, "s2-b64-16m_20*"))
    assert "seed2" not in str(harness.ckpt_path("s2_b64_16m", 25000))


def test_validation_split_uses_fixed_indices_when_present(tmp_path, monkeypatch):
    np.save(tmp_path / "val.npy", np.array([4, 0, 2], dtype=np.int64))
    monkeypatch.setattr("data.nlp.pajama_subset.RedPajamaSubsetDataset", lambda hp: list(range(6)))
    val = harness.validation_split({"fixed_val_indices": str(tmp_path / "val.npy"), "validation_split_pct": 0.5})
    assert [val[i] for i in range(len(val))] == [4, 0, 2]


def test_validation_split_falls_back_to_seeded_random_split(monkeypatch):
    monkeypatch.setattr("data.nlp.pajama_subset.RedPajamaSubsetDataset", lambda hp: list(range(100)))
    a = harness.validation_split({"validation_split_pct": 0.1})
    b = harness.validation_split({"validation_split_pct": 0.1})
    assert len(a) == 10 and list(a.indices) == list(b.indices)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd /home/quangnx/workspace/ebt/refs/EBT && PYTHONPATH=$PWD:$PWD/experiments/thinking_study/common /mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest experiments/thinking_study/common/test_harness.py -q`
Expected: the first three FAIL (`RUN_DIRS` holds strings, not tuples; the epoch glob is fixed to 0; no fixed-index branch). The fallback test passes.

- [ ] **Step 3: Implement**

Replace `CKPT_ROOT`/`RUN_DIRS`/`ckpt_path`/`validation_split` in `harness.py`:

```python
REPRO_CKPTS = REPO / "experiments/reproduce_xxs_scaling/runs/logs/checkpoints"
SCALE_CKPTS = REPO / "experiments/scale_to_16m_sequences/runs/logs/checkpoints"
RUN_DIRS = {  # key -> (checkpoint root, run-directory glob); trainer dirs are f"{run_name}_{timestamp}_"
    "tpp": (REPRO_CKPTS, "baseline-xxs-scaling-105k_2026-10-13_07-23-18_"),
    "s1": (REPRO_CKPTS, "ebt-xxs-scaling-105k_2026-10-13_09-09-28_"),
    "s2": (REPRO_CKPTS, "ebt-s2-xxs-scaling-105k_2026-10-14_05-20-41_"),
    "tpp_b64_16m": (SCALE_CKPTS, "tpp-b64-16m_20*"),
    "s2_b64_16m": (SCALE_CKPTS, "s2-b64-16m_20*"),
}


def ckpt_path(model: str, step: int = 105000) -> Path:
    root, pattern = RUN_DIRS[model]
    matches = sorted(p for d in root.glob(pattern) for p in d.glob(f"epoch=epoch=*-step=step={step}-*.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one checkpoint for {model} step {step}, found {matches}")
    return matches[0]


def validation_split(hparams: dict) -> Subset:
    """Validation set of a run: the fixed indices if the run used them, else the reproduction's seeded split."""
    from data.nlp.pajama_subset import RedPajamaSubsetDataset
    from data.fixed_val_split import fixed_val_split

    if hparams.get("fixed_val_indices"):
        return fixed_val_split(RedPajamaSubsetDataset(Namespace(**hparams)), hparams["fixed_val_indices"])[1]
    L.seed_everything(SPLIT_SEED, verbose=False)
    full = RedPajamaSubsetDataset(Namespace(**hparams))
    n_train = int(len(full) * (1 - hparams["validation_split_pct"]))
    return random_split(full, [n_train, len(full) - n_train])[1]
```

The monkeypatch in the test targets `data.nlp.pajama_subset.RedPajamaSubsetDataset`. That works because the import happens inside the function at call time.

In `run_q2.py`, replace the `--model` argument and the T++ load:

```python
    ap.add_argument("--model", choices=sorted(harness.RUN_DIRS), required=True)
    ap.add_argument("--tpp", choices=sorted(harness.RUN_DIRS), default="tpp", help="T++ run for off-trajectory candidates")
    ap.add_argument("--tpp-step", type=int, default=105000)
    ...
    tpp, _ = load_model(args.tpp, args.tpp_step)
```

Add `import harness` to its imports. In `run_q3.py`, use `ap.add_argument("--model", choices=sorted(harness.RUN_DIRS), required=True)` and add `import harness`.

- [ ] **Step 4: Run the tests, then the regression gate on a free GPU**

Run: `PYTHONPATH=$PWD:$PWD/experiments/thinking_study/common /mnt/data/conda-envs/quangnx/ebt312/bin/python -m pytest experiments/thinking_study/common/test_harness.py tests -q`
Expected: all pass.

Run: `cd experiments/thinking_study && CUDA_VISIBLE_DEVICES=<free> HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH=$OLDPWD:$PWD/common /mnt/data/conda-envs/quangnx/ebt312/bin/python common/gate.py`
Expected: `common/results/gate.json` reports every check passing, with the same numbers as commit `f9c4dc5` (T++ 3.69635; S1 3.83728 / 3.85633; S2 3.94320 / 6.16442).

- [ ] **Step 5: Commit**

```bash
git add experiments/thinking_study/common/harness.py experiments/thinking_study/common/test_harness.py \
        experiments/thinking_study/q2_verifier_quality/run_q2.py experiments/thinking_study/q3_energy_landscape/run_q3.py
git commit -m "feat: harness run registry for scale-16m runs, any-epoch checkpoints, fixed validation indices

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Validation indices, 16.1M-document dataset, verification

**Files:**
- Create: `experiments/scale_to_16m_sequences/export_val_indices.py`
- Create: `experiments/scale_to_16m_sequences/verify_dataset.py`
- Create (committed artifact): `experiments/scale_to_16m_sequences/val_indices_repro_33320.npy`

**Interfaces:**
- Consumes: `harness.validation_split`, `harness.ckpt_path` (Task 3).
- Produces: dataset at `$WS/data/full16m` (`--dataset_dir` for all runs). Validation file `$WS/val_indices_repro_33320.npy`.

- [ ] **Step 1: Export the reproduction's validation indices**

```python
# experiments/scale_to_16m_sequences/export_val_indices.py
"""Save the reproduction runs' 33,320 validation indices (seeded random_split over the 3.4M-doc subset)."""
from pathlib import Path

import numpy as np
import torch

from harness import ckpt_path, validation_split

OUT = Path(__file__).resolve().parent / "val_indices_repro_33320.npy"

hparams = torch.load(ckpt_path("s2"), map_location="cpu", weights_only=False)["hyper_parameters"]
val = validation_split(dict(hparams))
idx = np.asarray(val.indices, dtype=np.int64)
assert len(idx) == 33320, len(idx)
np.save(OUT, idx)
print(f"wrote {OUT}: {len(idx)} indices, max {idx.max()}")
```

Run: `cd /home/quangnx/workspace/ebt/refs/EBT && HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH=$PWD:$PWD/experiments/thinking_study/common /mnt/data/conda-envs/quangnx/ebt312/bin/python experiments/scale_to_16m_sequences/export_val_indices.py`
Expected: `wrote .../val_indices_repro_33320.npy: 33320 indices, max <3399999`.

- [ ] **Step 2: Build the dataset, reusing the downloaded shards (CPU only, ~3–4 h, run in the background)**

```bash
WS=experiments/scale_to_16m_sequences
mkdir -p $WS/data/full16m/pajama_subset/raw_cache
cp -al experiments/reproduce_xxs_scaling/data/full/pajama_subset/raw_cache/documents $WS/data/full16m/pajama_subset/raw_cache/
mkdir -p $WS/logs
HF_HOME=/mnt/data/quangnx/huggingface nohup /mnt/data/conda-envs/quangnx/ebt312/bin/python -m data.nlp.pajama_subset \
  --dataset_dir $WS/data/full16m --num_docs 16100000 --download_workers 16 > $WS/logs/build_full16m.log 2>&1 &
```
`cp -al` hard-links the 383 cached shards (same filesystem, no extra space). Expected end of log: `wrote ...; missing shards: 0`. If any shards are missing, verification in Step 4 decides whether that matters.

- [ ] **Step 3: Write the verification script**

```python
# experiments/scale_to_16m_sequences/verify_dataset.py
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
        "ids prefix equal": all(np.array_equal(old_ids[s:s + CHUNK], new_ids[s:s + CHUNK]) for s in range(0, n_old, CHUNK)),
        "val indices inside old prefix": bool(val.max() < n_old),
        f"train rows >= {MIN_TRAIN}": n_new - len(val) >= MIN_TRAIN,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"old {n_old:,} docs, new {n_new:,} docs, train {n_new - len(val):,}, val {len(val):,}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
```

Validation order (Review Focus 5) follows from the equal prefix: position i of the fixed validation set is old row `val[i]` in both builds.

- [ ] **Step 4: Run the verification after the build finishes**

Run: `/mnt/data/conda-envs/quangnx/ebt312/bin/python experiments/scale_to_16m_sequences/verify_dataset.py`
Expected: four `PASS` lines and `train 16,066,680, val 33,320`. Any `FAIL` stops the plan; report it to the user.

- [ ] **Step 5: Commit**

```bash
git add experiments/scale_to_16m_sequences/{export_val_indices.py,verify_dataset.py,val_indices_repro_33320.npy}
git commit -m "feat: export reproduction validation indices; verify 16.1M-doc superset dataset

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: DDP launch scripts, smoke test, benchmark (decision gate)

**Files:**
- Create: `experiments/scale_to_16m_sequences/common.sh`, `run_tpp.sh`, `run_s2.sh`, `benchmark_ddp.py`

**Interfaces:**
- Consumes: the dataset and indices (Task 4), `--fixed_val_indices` (Task 2).
- Produces: `GPU_IDS=a,b bash run_tpp.sh|run_s2.sh --run_name NAME --max_steps N --val_check_interval M [extra args]`. Checkpoints go to `$WS/runs/logs/checkpoints/NAME_<timestamp>_/`; CSV metrics to `$WS/runs/logs/csv/NAME/version_0/metrics.csv`.

- [ ] **Step 1: Write the launch scripts**

```bash
# experiments/scale_to_16m_sequences/common.sh
#!/usr/bin/env bash
# Shared settings: xxs, paper hparams as in ../reproduce_xxs_scaling/common.sh, but DDP on 2 GPUs (global batch 64),
# a 16.1M-doc superset dataset, and the reproduction's fixed 33,320-doc validation set.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$WS/../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface
mkdir -p "$WS/logs" "$WS/runs"

: "${GPU_IDS:?set GPU_IDS to two idle GPUs on one NUMA node, e.g. 4,5}"
for id in ${GPU_IDS//,/ }; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$id")
  if [ "$used" -gt 1000 ] && [ "${ALLOW_BUSY_GPU:-0}" != 1 ]; then echo "GPU $id uses ${used} MiB" >&2; exit 1; fi
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TRACKIO_DIR="$WS/runs/trackio"
cd "$WS/runs"  # train_model.py writes ./logs/{checkpoints,csv}

COMMON_ARGS=(
  --modality NLP --model_size xxs
  --pretokenize_dataset --tokenizer EleutherAI/gpt-neox-20b
  --context_length 256 --ffn_dim_multiplier 1
  --gpus 2 --distributed_strategy ddp
  --peak_learning_rate 0.0012 --batch_size_per_device 32 --accumulate_grad_batches 1
  --gradient_clip_val 1.0 --weight_decay 0.01 --min_lr_scale 10 --warm_up_steps 10000
  --dataset_name pajama_subset --dataset_dir "$WS/data/full16m"
  --fixed_val_indices "$WS/val_indices_repro_33320.npy" --validation_split_pct 0.0098
  --num_workers 12 --save_top_k_ckpts -1
  --wandb_project ebt_scale16m
  --log_model_archi --log_gradients
  --set_matmul_precision medium
)
```

```bash
# experiments/scale_to_16m_sequences/run_tpp.sh
#!/usr/bin/env bash
# Transformer++ xxs; caller passes --run_name, --max_steps, --val_check_interval (max_scheduling_steps follows max_steps).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" "$REPO/train_model.py" --model_name baseline_transformer "${COMMON_ARGS[@]}" "$@"
```

```bash
# experiments/scale_to_16m_sequences/run_s2.sh
#!/usr/bin/env bash
# EBT System 2, MCMC hparams from paper Table D.4 (identical to ../reproduce_xxs_scaling/run_ebt_s2_xxs.sh).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$PY" "$REPO/train_model.py" --model_name ebt "${COMMON_ARGS[@]}" \
  --normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise \
  --no_mcmc_detach --mcmc_replay_buffer --truncate_mcmc --langevin_dynamics_noise 3.0 \
  --mcmc_step_size 5 --randomize_mcmc_step_size_scale 2 \
  --mcmc_num_steps 1 --randomize_mcmc_num_steps 2 --randomize_mcmc_num_steps_min 2 \
  "$@"
```

`--model_name baseline_transformer` matches `../reproduce_xxs_scaling/run_baseline_xxs.sh`. `max_scheduling_steps` defaults to `max_steps` (`train_model.py`, `if args.max_scheduling_steps == -1`), so it is not passed.

- [ ] **Step 2: DDP smoke test (30 steps each, real data, small validation)**

```bash
cd /home/quangnx/workspace/ebt/refs/EBT/experiments/scale_to_16m_sequences
for m in tpp s2; do
  GPU_IDS=4,5 bash run_$m.sh --run_name smoke-$m --wandb_project ebt_scale16m_smoke \
    --max_steps 30 --val_check_interval 10 --limit_val_batches 5 > logs/smoke_$m.log 2>&1
done
grep -h "effective_batch_size" logs/smoke_*.log
/mnt/data/conda-envs/quangnx/ebt312/bin/python -c "
import glob, torch
for p in sorted(glob.glob('runs/logs/checkpoints/smoke-*/last.ckpt')):
    print(p, torch.load(p, map_location='cpu', weights_only=False)['hyper_parameters']['fixed_val_indices'])"
```
Expected: both exit 0; `effective_batch_size: 64` in both logs; each `last.ckpt` prints the path of `val_indices_repro_33320.npy`. Delete the smoke checkpoints afterwards: `rm -r runs/logs/checkpoints/smoke-*`.

- [ ] **Step 3: Benchmark script**

```python
# experiments/scale_to_16m_sequences/benchmark_ddp.py
"""Steady-state DDP step time: (t_350 - t_50) / 300 steps, then ETAs for the planned step counts."""
import json
import os
import subprocess
import sys
import time

WS = os.path.dirname(os.path.abspath(__file__))
PLANNED = {"16m": 250_000}


def timed_run(model: str, steps: int) -> float:
    cmd = ["bash", os.path.join(WS, f"run_{model}.sh"), "--run_name", f"bench-{model}-{steps}",
           "--wandb_project", "ebt_scale16m_bench", "--max_steps", str(steps),
           "--limit_val_batches", "0", "--val_check_interval", "100", "--save_top_k_ckpts", "0"]
    start = time.perf_counter()
    with open(os.path.join(WS, "logs", f"bench_{model}_{steps}.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def main(model: str) -> None:
    sps = 300 / (timed_run(model, 350) - timed_run(model, 50))
    result = {"model": model, "steps_per_s": round(sps, 3), **{f"eta_h_{k}": round(v / sps / 3600, 2) for k, v in PLANNED.items()}}
    with open(os.path.join(WS, "logs", f"benchmark_{model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
```

Validation is excluded (`--limit_val_batches 0`). With the planned validation cadence, full validation adds roughly 5–10% (the 105k reproduction averaged 2.79 steps/s, against a 3.0 steps/s benchmark).

- [ ] **Step 4: Run the benchmark and apply the decision gate**

Run: `for m in tpp s2; do GPU_IDS=4,5 /mnt/data/conda-envs/quangnx/ebt312/bin/python benchmark_ddp.py $m; done`
Expected (from the 1-GPU numbers and 70–98% DDP efficiency): S2 ≈ 2.6–2.75 steps/s (`eta_h_16m` ≈ 25–27); T++ ≈ 11.5–15 steps/s (`eta_h_16m` ≈ 4.6–6).
**Gate:** if S2's `eta_h_16m` > 32, or T++'s > 8, stop and report the numbers to the user before Task 6. Clean up: `rm -r runs/logs/checkpoints/bench-*`.

- [ ] **Step 5: Commit**

```bash
git add common.sh run_tpp.sh run_s2.sh benchmark_ddp.py logs/benchmark_*.json
git commit -m "feat: DDP 2-GPU launch scripts and benchmark for scale-16m runs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

(`logs/*.log` is git-ignored; the benchmark JSONs are committed.)

### Task 6: Training queue (T++ then S2, 250k steps each)

**Files:**
- Create: `experiments/scale_to_16m_sequences/run_all.sh`

**Interfaces:**
- Consumes: `run_tpp.sh`, `run_s2.sh` (Task 5).
- Produces: runs `tpp-b64-16m`, `s2-b64-16m` under `runs/logs/{checkpoints,csv}`.

- [ ] **Step 1: Write the queue**

```bash
# experiments/scale_to_16m_sequences/run_all.sh
#!/usr/bin/env bash
# Sequential DDP queue on GPU_IDS: 250k x 64 = 16M seqs per run, validate (and checkpoint) every 25k.
# T++ first: it is ~5x cheaper and checks the pipeline end to end before the long S2 run.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${GPU_IDS:?set GPU_IDS}"
run() {  # model run_name max_steps val_interval [extra]
  local model=$1 name=$2 steps=$3 every=$4; shift 4
  echo "=== $name start $(date -Is)"
  GPU_IDS=$GPU_IDS bash "$WS/run_$model.sh" --run_name "$name" --max_steps "$steps" --val_check_interval "$every" "$@" \
    > "$WS/logs/$name.log" 2>&1
  echo "=== $name end $(date -Is)"
}
run tpp tpp-b64-16m 250000 25000
run s2 s2-b64-16m 250000 25000
echo "=== all done $(date -Is)"
```

- [ ] **Step 2: Launch in the background**

```bash
cd /home/quangnx/workspace/ebt/refs/EBT/experiments/scale_to_16m_sequences
GPU_IDS=4,5 nohup bash run_all.sh > logs/run_all.log 2>&1 &
```
Expected: `logs/run_all.log` shows `=== tpp-b64-16m start`. Within ~2 min, `logs/tpp-b64-16m.log` shows training steps. If a run crashes, resume it with the same command plus `--resume_training_ckpt <abs path>/last.ckpt`.

- [ ] **Step 3: Check the T++ run before S2 gets far**

After `tpp-b64-16m` ends (~5–6 h): `ls runs/logs/checkpoints/tpp-b64-16m_*/` must list 10 step checkpoints (25000 … 250000) plus `last.ckpt`. `runs/logs/csv/tpp-b64-16m/version_0/metrics.csv` must have 10 `valid_loss` rows. The final value should be below 3.6964 (the 105k/b32 T++ on the same validation set). If it is not, stop the queue and report before S2 runs for a day.

- [ ] **Step 4: Commit the queue script**

```bash
git add run_all.sh && git commit -m "feat: sequential DDP training queue for scale-16m runs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 7: Evaluation sweep (Q2, Q3) on the new checkpoints

**Files:**
- Create: `experiments/scale_to_16m_sequences/run_eval.sh`

**Interfaces:**
- Consumes: harness keys (Task 3), checkpoints (Task 6), `run_q2.py`, `analyze_q2.py`, `run_q3.py`, `analyze_q3.py`.
- Produces: `results/q2/<key>_step<N>_noise<σ>/metrics.json`, `results/q3/<key>_step<N>/metrics.json`.

- [ ] **Step 1: Write the sweep**

```bash
# experiments/scale_to_16m_sequences/run_eval.sh
#!/usr/bin/env bash
# Q2 (verifier audit) and Q3 (landscape) on scale-16m S2 checkpoints, same eval subset as the thinking study.
# PART=final|curve: final = the 250k checkpoint (Q2 at sigma 3 and 0, Q3); curve = Q2 sigma 3 at 50k-200k.
# The 3.36M point is the thinking study's existing s2 105k results (same eval subset).
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$WS/../.." && pwd)"
TS="$REPO/experiments/thinking_study"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
cd "$REPO"
q2() {  # key step noise tpp_key tpp_step
  local out="$WS/results/q2/${1}_step${2}_noise${3}"
  $PY "$TS/q2_verifier_quality/run_q2.py" --model "$1" --step "$2" --noise "$3" --tpp "$4" --tpp-step "$5" --out-dir "$out"
  $PY "$TS/q2_verifier_quality/analyze_q2.py" "$out" > "$out/metrics.log"
}
q3() {  # key step
  local out="$WS/results/q3/${1}_step${2}"
  $PY "$TS/q3_energy_landscape/run_q3.py" --model "$1" --step "$2" --out-dir "$out"
  $PY "$TS/q3_energy_landscape/analyze_q3.py" "$out"
}
case "${PART:?set PART=final|curve}" in
  final)
    for noise in 3 0; do q2 s2_b64_16m 250000 "$noise" tpp_b64_16m 250000; done
    q3 s2_b64_16m 250000 ;;
  curve)
    for step in 50000 100000 150000 200000; do q2 s2_b64_16m "$step" 3 tpp_b64_16m "$step"; done ;;
esac
echo "=== done $(date -Is)"
```

`run_q3.py` takes the Langevin noise values from the checkpoint hparams (0 and 3 for S2), as in the thinking study.

- [ ] **Step 2: Run on the two GPUs in parallel once training is done**

```bash
cd /home/quangnx/workspace/ebt/refs/EBT/experiments/scale_to_16m_sequences
GPU_ID=4 PART=final nohup bash run_eval.sh > logs/eval_final.log 2>&1 &
GPU_ID=5 PART=curve nohup bash run_eval.sh > logs/eval_curve.log 2>&1 &
```
Expected: both logs end with `=== done`. Each Q2 result directory has `metrics.json`. Wall time ≈ 40 min (`final`: 2 Q2 × ~6 min + 1 Q3 × ~25 min) and ≈ 30 min (`curve`).

- [ ] **Step 3: Sanity check against the logged validation loss**

The Q2 single-sample CE (`single_sample.ce`, 1,024-sequence subset, σ = 3) at the final checkpoint should be within about 0.05 nats of the run's logged `valid_final_step_loss` (full 33,320 docs). Both use the training noise; the subset differs, as in the thinking study (3.9224 vs 3.9431 for the 105k S2). A larger gap means checkpoint loading or the split is wrong; stop and investigate.

- [ ] **Step 4: Commit**

```bash
git add run_eval.sh && git commit -m "feat: Q2/Q3 evaluation sweep for scale-16m checkpoints

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 8: Collect results and write the report

**Files:**
- Create: `experiments/scale_to_16m_sequences/collect.py`
- Create: `experiments/scale_to_16m_sequences/report/result.md` (generated), `report/report.md` (written)

**Interfaces:**
- Consumes: `read_valid_rows`, `comparable_loss` from `../reproduce_xxs_scaling/collect_results.py`; the CSV logs; `results/q2|q3/*/metrics.json`; and the thinking study's `q2_verifier_quality/results/s2_step105000_noise{3,0}` and `q3_energy_landscape/results/s2_step105000` (the 3.36M/b32 point).

- [ ] **Step 1: Write the collector**

```python
# experiments/scale_to_16m_sequences/collect.py
"""Tables for report/result.md: validation curves and S2-T++ gaps per data point, plus Q2/Q3 headline metrics."""
import glob
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent
REPRO = WS.parent / "reproduce_xxs_scaling"
TS = WS.parent / "thinking_study"
sys.path.insert(0, str(REPRO))
from collect_results import comparable_loss, read_valid_rows  # noqa: E402

POINTS = {  # label -> (T++ csv glob, S2 csv glob, Q2 dir σ=3, Q3 dir)
    "3.36M seqs, b32 (105k)": (REPRO / "runs/logs/csv/baseline-xxs-scaling-105k", REPRO / "runs/logs/csv/ebt-s2-xxs-scaling-105k",
                              TS / "q2_verifier_quality/results/s2_step105000_noise3", TS / "q3_energy_landscape/results/s2_step105000"),
    "16M seqs, b64 (250k)": (WS / "runs/logs/csv/tpp-b64-16m", WS / "runs/logs/csv/s2-b64-16m",
                            WS / "results/q2/s2_b64_16m_step250000_noise3", WS / "results/q3/s2_b64_16m_step250000"),
}


def curve(run_dir: Path) -> dict[int, float]:
    return {r["step"]: comparable_loss(r) for r in read_valid_rows(sorted(glob.glob(str(run_dir / "version_*/metrics.csv"))))}


def main() -> None:
    lines = ["## Final validation loss and gap", "",
             "| point | T++ | S2 | gap (S2 − T++) | Q2 BoN-32 efficiency (σ=3) | Q2 consistency (gap>0.1) | Q3 CE trained k → k=32 |",
             "|---|---|---|---|---|---|---|"]
    for label, (tpp_dir, s2_dir, q2_dir, q3_dir) in POINTS.items():
        tpp, s2 = curve(tpp_dir), curve(s2_dir)
        last = max(s2)
        q2 = json.loads((q2_dir / "metrics.json").read_text())["e_final"]
        q3 = json.loads((q3_dir / "metrics.json").read_text())["part_a_noise0"]
        ce_k = q3["ce_after_k_steps"]
        lines.append(f"| {label} | {tpp[last]:.4f} | {s2[last]:.4f} | {s2[last] - tpp[last]:+.4f} | "
                     f"{q2['best_of_n']['32']['verifier_efficiency']:+.2f} | {q2['pairwise_consistency_gap_gt_0.1'][0]:.3f} | "
                     f"{q3['best_k']['ce_at_train_steps']:.3f} → {ce_k[str(max(map(int, ce_k)))]:.3f} |")
    tpp, s2 = curve(POINTS["16M seqs, b64 (250k)"][0]), curve(POINTS["16M seqs, b64 (250k)"][1])
    lines += ["", "## Gap along the 16M run", "", "| step | T++ | S2 | gap |", "|---|---|---|---|"]
    lines += [f"| {k} | {tpp[k]:.4f} | {s2[k]:.4f} | {s2[k] - tpp[k]:+.4f} |" for k in sorted(set(tpp) & set(s2))]
    out = WS / "report/result.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
```

JSON keys follow `analyze_q2.py`/`analyze_q3.py`. `json.dump` turns integer dict keys into strings, which is why the code indexes `'32'` and `str(k)`. `pairwise_consistency` returns `(share, n_pairs)`, so the code takes `[0]`.

- [ ] **Step 2: Run it**

Run: `/mnt/data/conda-envs/quangnx/ebt312/bin/python experiments/scale_to_16m_sequences/collect.py`
Expected: two rows in the first table and 10 rows in the curve table; no `KeyError`.

- [ ] **Step 3: Write `report/report.md`**

Write the report in the style of `../thinking_study/q2_verifier_quality/report.md`:
- frontmatter and a 3-bullet TL;DR;
- §1 setup (link this plan);
- §2 results, linking `result.md` and adding the Q2 curve (efficiency and consistency at 50k–250k);
- §3 verdict per pre-registered claim (§0): supported / not supported / inconclusive, with the threshold used;
- §4 limitations: one seed per point (seed variance unmeasured); the two points differ in batch size (32 vs 64; at a fixed number of sequences the smaller batch takes more steps, which affects both models alike); xxs only; batch 64 vs the paper's 128; data still 8× below the paper.

- [ ] **Step 4: Commit**

```bash
git add experiments/scale_to_16m_sequences/collect.py experiments/scale_to_16m_sequences/report/
git commit -m "docs: scale-16m results and report

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

## Timeline (2 GPUs, `4,5`)

| Phase | Resource | Estimate |
|---|---|---|
| Tasks 1–3 (code, tests, gate) | 1 GPU for the gate (~10 min) | ~1 h, in parallel with the data build |
| Task 4 data build | CPU, network | 3–4 h |
| Task 5 smoke and benchmark | 2 GPUs | ~0.5 h |
| T++ 16M / S2 16M | 2 GPUs | 4.6–6 h / 25–27 h |
| Task 7 eval | 2 GPUs in parallel | ~0.7 h |
| Total wall-clock | | **≈ 32–35 h** |

Validation time is included approximately (5–10%). The benchmark gate in Task 5 replaces these estimates with measured ones.
