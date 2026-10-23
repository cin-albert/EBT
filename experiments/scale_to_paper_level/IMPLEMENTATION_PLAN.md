---
title: "Scale S2 vs Transformer++ to the paper's data budget — implementation plan"
date: 2026-10-23
tags: [ebt, scaling, system2, plan, reproduction]
status: plan
---

# Scale S2 vs Transformer++ to the paper's data budget — Implementation Plan

> [!summary] TL;DR
> - Train EBT-S2 and Transformer++ (xxs) at the paper's thinking-model scale: **global batch 128 × 1M steps = 128M sequences** (≈ 32.8B tokens), on the **full RedPajama-V2 `sample-100B`** (≈ 66M documents in the paper, so ≈ 1.9 epochs), with a **new random 33,000-document validation set** drawn from the full build.
> - Code work is small: multi-epoch resume, validation by global step across epoch boundaries, a new validation-index file, run keys for the eval harness, and one collector for the paper's figures. Everything else reuses `scale_to_16m_sequences` and `thinking_study`.
> - Seven pre-registered claims (§0.2) against Table 3, Fig. 6a, 6b, 7 and B.1a. Everything runs on **GPUs 0–3 only** (4 and 5 are reserved for other users). The runs are sequential: T++ first (≈ 1 day), then S2 (≈ 4–4.5 days). Total ≈ 7 days, including ≈ 12 h of data build and ≈ 1 day of evaluation.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the paper's language thinking results (Table 3, Fig. 6a/6b/7, B.1a) at its stated data budget, and extend our 3.36M → 16M series with a 128M point.

**Architecture:**
- Reuse the repo trainer with two changes: resume at any epoch (`data/resume_order.py`) and global-step validation (`--check_val_every_n_epoch 0`).
- Reuse `data.nlp.pajama_subset` to build all 8,400 `sample-100B` shards.
- Reuse the thinking-study harness and the Q2/Q3/OOD scripts with new run keys. New code lives in `experiments/scale_to_paper_level/` (`WS` below); heavy data and checkpoints stay in gitignored `$WS/data` and `$WS/runs`.

**Tech Stack:** PyTorch 2.4 + Lightning 2.4 (DDP), env `/home/quangnx/miniconda3/envs/ebt312`, numpy, pytest, matplotlib.

**Spec:** §0 below, decided in discussion on 2026-10-23. Context: [[report|scale-16m report]] (`../scale_to_16m_sequences/report/report.md`), OOD report (`../scale_to_16m_sequences/ood/report/report.md`), [[SUMMARIZED_FINDINGS]], paper App. D.1.2 and Table D.4.

## 0. Spec: decisions and rationale

| Decision | Value | Why |
|---|---|---|
| Models | T++ and EBT-S2 only, xxs (6 layers, D = 384) | A1: App. D.1.2 text says xxs (Table D.4 says "Small"); matches all earlier runs |
| S2 hparams | Identical to `../scale_to_16m_sequences/run_s2.sh` (Table D.4) | Comparability with the 3.36M and 16M points |
| Batch / steps | Global batch 128 × 1,000,000 steps; LR 0.0012, warmup 10k, cosine to 1M, `min_lr_scale 10` | Paper App. D.1.2 / Table D.4 |
| Parallelism | Both models: DDP on GPUs 0–3, 4 × 32, `--accumulate_grad_batches 1`. Sequential: T++ then S2 | Only GPUs 0–3 may be used. S2 at 32/GPU fits 80 GB (≈ 61 GB peak at batch 32 on the 94 GB card). T++ first: it is short and tests the full pipeline (epoch boundary, validation cadence, checkpoints) before the 4-day S2 run |
| Data | All 8,400 `sample-100B` shards, shard order seed 33, first 257 tokens per document | A2: paper's corpus; it has ≈ 66M documents, so 128M sequences ≈ 1.9 epochs, as in the paper |
| Validation | **New** 33,000 indices drawn uniformly (numpy seed 33) from the full build, excluded from training | A3: paper-style random split; replaces the old 33,320 set |
| Ablations (Table 2) | Not run | A4: each variant costs a full S2 run |
| Validation / checkpoints | Every 50k global steps (20 per run), keep all | Fig. 6b / B.1a need curves over training (paper's x-axis reaches ≈ 33B tokens = 1M × 128 × 256) |
| Validation semantics | Repo default (S2: Langevin σ = 3 and k = 3 at validation) | Same as earlier runs; the paper does not state its own |
| Seeds | One per run | As before; thresholds are fixed in advance instead |

### 0.1 Definitions used by every claim

- **ppl** of a set = exp(mean token CE over all non-pad tokens), as `get_ppl` and the OOD report.
- **% ppl improvement** of variant over base = 100 · (1 − exp(CE_variant − CE_base)) (existing `collect_ood.pct_improvement`).
- **No Thinking / Thinking Longer** = S2 single sample at k = 2 / k = 3 steps (Q3 Part A). **Self-Verification BoN-N** = lowest final energy among N candidates of 3 steps (Q2), so 3N forward passes. All at σ = 3 unless stated; σ = 0 is reported as a secondary row.
- **Max Thinking** = BoN-10 (30 passes, the rightmost point of Fig. 6a).
- **OOD sets** = GSM8K, SQuAD, BB Elementary Math QA, BB Dyck (as the OOD report); **shift** = S2 ppl(set) / S2 ppl(validation).
- **Checkpoint for Table 3** = the saved checkpoint with the lowest `valid_final_step_loss` (S2) / `valid_loss` (T++), the paper's "best checkpoint".

### 0.2 Pre-registered claims

Written before training; `collect_paper.py` (Task 7) prints each verdict.

| # | Paper claim | Supported if | Paper value |
|---|---|---|---|
| C1 | S2 − T++ gap keeps shrinking with data | validation CE gap at the Table 3 checkpoint < 0.179 − 0.03 = **0.149**; "matches paper" if ≤ **0.114** | +0.064 (ppl 33.43 vs 31.36) |
| C2 | Thinking Longer lowers OOD ppl (Fig. 6a) | mean OOD ppl at k = 3 is lower than at k = 2 by > **10%** | ≈ 20% (44.8 → 35.9) |
| C3 | Self-Verification lowers it further (Fig. 6a) | mean OOD ppl at BoN-10 is lower than at k = 3 by > **2%** | ≈ 11% (35.9 → 31.8) |
| C4 | Verification gain grows with training (Fig. 6b, Dyck) | Dyck % ppl improvement of BoN-5 over a single 3-step sample is > 0 at the final checkpoint **and** Spearman(tokens, improvement) > 0 over the 20 checkpoints | 4–8% → 10–14% |
| C5 | Fewer adversarial samples with training (Fig. B.1a) | on validation, % improvement of BoN-10 over BoN-2 is > 0 at the final checkpoint **and** Spearman(tokens, improvement) > 0 | qualitative |
| C6 | Thinking gain grows with OOD shift (Fig. 7) | Spearman(shift, Max-Thinking % improvement over No Thinking) > 0 over 5 points (validation + 4 OOD sets) | linear, 12% → 23% |
| C7 | EBT beats T++ downstream despite worse pretraining (Table 3) | S2 ppl < T++ ppl on ≥ **3 of 4** OOD sets (S2 single sample, k = 3, σ = 3) | 3 of 4 |

C1 compares a gap on the new validation set with the 16M gap on the old one. Both sets are random RedPajama documents, so the gap is comparable, but the absolute losses are not.

## Global Constraints

- Python: `/home/quangnx/miniconda3/envs/ebt312/bin/python` (variable `PY` in `common.sh`); `HF_HOME=$HOME/.cache/huggingface` unless set; `TMPDIR=$WS/tmp`.
- Machine: 6 × H100 80GB HBM3, 128 CPUs, 2 TB RAM, ≈ 23 TB free on `/`. **Only GPUs 0, 1, 2, 3 may be used**, for training, benchmarks, tests and evaluation; GPUs 4 and 5 belong to other users. Every launch script refuses a `GPU_IDS`/`GPU_ID` outside {0, 1, 2, 3}, a requested GPU using > 1000 MiB, or `/` with < 500 GB free.
- Fixed values: context 256, `--ffn_dim_multiplier 1`, LR 0.0012, warmup 10,000, `--max_steps 1000000 --max_scheduling_steps 1000000`, weight decay 0.01, grad clip 1.0, `--set_matmul_precision medium`, seed 33, validation file `$WS/val_indices_paper_33000.npy` (committed).
- Branch `scale-paper` from the current `exp/xxs-scaling` HEAD. Commit locally per task; **never push**.
- Every file is in English. Markdown is Obsidian-friendly with a 3-bullet TL;DR. Python follows the `python-guidelines-research` skill.
- Do not modify `../reproduce_xxs_scaling/` or `../scale_to_16m_sequences/` except to import from them; changes to shared code (`data/`, `train_model.py`, `base_model_trainer.py`, `../thinking_study/common/harness.py`) must keep their existing tests passing.

## Review Focus

1. **Resume inside epoch 1.** Lightning sets the sampler epoch to 1, so the remaining order uses seed 33 + 1. A resume that rebuilds epoch-0 order would silently repeat epoch-0 documents. Task 3 tests `resume_position` and `continue_epoch_order(epoch=1)` against a real `DistributedSampler`.
2. **Validation cadence across the epoch boundary.** With an integer `val_check_interval` and `check_val_every_n_epoch=1`, Lightning counts batches *within* an epoch, so after ≈ 516k steps validation drifts off the 50k grid. Task 3's smoke run crosses an epoch boundary and asserts validation and checkpoint steps are exact multiples of the interval.
3. **Validation documents leaking into training.** The new indices must be unique, in range, and absent from the train subset; `fixed_val_split` already checks this, and Task 2 asserts disjointness on the real build.
4. **The build stops early or is unexpectedly small.** Shards that 404 are recorded in `meta.json` `tags_missing`. Task 1 fails if more than 1% of tags are missing or if fewer than 50M documents are built, rather than training on a silently smaller corpus.
5. **Old checkpoints and earlier evals must behave as before.** New harness keys only add entries; Task 4 reruns `../thinking_study/common/test_harness.py` and the `tests/` suite.

---

## File structure

| Path | Responsibility | New / modified |
|---|---|---|
| `data/resume_order.py` | `resume_position(...)`; `continue_epoch_order(..., epoch=0)` | Modified |
| `base_model_trainer.py` (`setup`, l. 483–486) | Use `resume_position` so any epoch can be resumed | Modified |
| `train_model.py` (l. 258–281, 594, 609) | `--check_val_every_n_epoch 0` → `None`; integer `val_check_interval`; flag help text | Modified |
| `tests/test_resume_order.py` | Multi-epoch resume tests | Modified |
| `tests/test_val_cadence.py` | Smoke run across an epoch boundary | New |
| `experiments/thinking_study/common/harness.py` | Run keys `tpp_paper`, `s2_paper` | Modified |
| `$WS/.gitignore` | `data`, `runs`, `results`, `tmp`, `logs/*.log`, `*.npz`, `__pycache__/` | New |
| `$WS/common.sh`, `run_tpp.sh`, `run_s2.sh` | Env, checks and DDP launch | New |
| `$WS/benchmark.py` | Copy of `../scale_to_16m_sequences/benchmark_ddp.py` with `PLANNED = {"paper": 1_000_000}` and this workspace's run scripts | New |
| `$WS/make_val_indices.py`, `$WS/test_make_val_indices.py` | Write and test `val_indices_paper_33000.npy` | New |
| `$WS/build_data.sh` | Full-corpus build and its acceptance checks | New |
| `$WS/run_eval.sh` | Table 3 ppl, Q2/Q3 per set, checkpoint curves | New |
| `$WS/collect_paper.py`, `$WS/test_collect_paper.py` | C1–C7 verdicts, `report/result.md`, figures | New |
| `$WS/report/report.md` | Findings | New |

---

### Task 1: Workspace, environment and full-corpus data build

**Files:**
- Create: `$WS/.gitignore`, `$WS/common.sh`, `$WS/build_data.sh`

**Interfaces:**
- Produces: `common.sh` exporting `WS`, `REPO`, `PY`, `HF_HOME`, `TMPDIR`, `DATA_DIR=$WS/data/full`, and the array `COMMON_ARGS` (as `../scale_to_16m_sequences/common.sh`, with `--gpus ${#GPUS}`, `--dataset_dir $DATA_DIR`, `--fixed_val_indices $WS/val_indices_paper_33000.npy`, `--val_check_interval 50000 --check_val_every_n_epoch 0`, `--max_steps 1000000 --max_scheduling_steps 1000000`, `--save_top_k_ckpts -1`, `--wandb_project ebt_scale_paper`). The allowed-GPU check (ids ⊆ {0, 1, 2, 3}), the GPU-idle check and the ≥ 500 GB disk check run before anything else.
- Produces: `$DATA_DIR/pajama_subset/EleutherAI_gpt-neox-20b/max_length_257/{input_ids.npy,lengths.npy,meta.json}`.

- [ ] **Step 1: Create branch `scale-paper`, the `.gitignore` and `common.sh`; confirm the env works**

Run: `cd $REPO && $PY -m pytest tests/ -q`
Expected: all pass (as on the old machine).

- [ ] **Step 2: Smoke build (20,000 docs) to `$WS/data/smoke`**

Run: `bash $WS/build_data.sh smoke`
Expected: `meta.json` with `num_docs: 20000`.

- [ ] **Step 3: Full build**

`build_data.sh full` runs `python -m data.nlp.pajama_subset --dataset_dir $DATA_DIR --num_docs 80000000 --download_workers 32` (capacity above any plausible count; the builder stops when the 8,400 tags run out) and logs to `$WS/logs/build_full.log`. Expected duration ≈ 10–14 h (the 3.4M build took 42 min).

- [ ] **Step 4: Acceptance checks, appended to `build_data.sh full`**

Fail if `len(tags_missing) > 84` (1% of 8,400) or `num_docs < 50_000_000`. Print `num_docs`, the language mix and `num_docs / 66e6`. Record the numbers in `$WS/logs/data_stats.json`.

- [ ] **Step 5: Commit** `.gitignore`, `common.sh`, `build_data.sh`, `logs/data_stats.json`.

### Task 2: New validation indices

**Files:**
- Create: `$WS/make_val_indices.py`, `$WS/test_make_val_indices.py`, `$WS/val_indices_paper_33000.npy`

**Interfaces:**
- Produces: `make_val_indices(n_docs: int, n_val: int = 33_000, seed: int = 33) -> np.ndarray` (sorted, unique, int64, all `< n_docs`); the CLI writes the `.npy` from `meta.json` `num_docs`.

- [ ] **Step 1: Write the failing tests**

```python
def test_val_indices_unique_sorted_in_range():
    idx = make_val_indices(1_000_000)
    assert len(idx) == 33_000 and idx.dtype == np.int64
    assert np.all(np.diff(idx) > 0) and idx[0] >= 0 and idx[-1] < 1_000_000

def test_val_indices_deterministic():
    assert np.array_equal(make_val_indices(10**6), make_val_indices(10**6))

def test_val_indices_cover_whole_range():  # not clustered in the first shards
    idx = make_val_indices(1_000_000)
    assert (idx >= 500_000).mean() > 0.45
```

- [ ] **Step 2: Run and confirm they fail.** `$PY -m pytest $WS/test_make_val_indices.py -v` → FAIL (import error).
- [ ] **Step 3: Implement** with `np.random.default_rng(seed).choice(n_docs, n_val, replace=False)`, then sort.
- [ ] **Step 4: Run the tests** → PASS. Then write the file from the full build and check disjointness on the real data: `fixed_val_split(RedPajamaSubsetDataset(...), path)` returns train + val lengths summing to `num_docs` with no shared index.
- [ ] **Step 5: Commit** the script, the test and the `.npy` (≈ 264 KB).

### Task 3: Multi-epoch resume and global-step validation

**Files:**
- Modify: `data/resume_order.py`, `base_model_trainer.py:483-486`, `train_model.py:258-281,594,609`
- Modify: `tests/test_resume_order.py`; Create: `tests/test_val_cadence.py`

**Interfaces:**
- Produces: `resume_position(global_steps: int, n_train: int, per_device_batch: int, world_size: int) -> tuple[int, int]` returning `(epoch, offset_samples)`, where steps per epoch = `ceil(ceil(n_train / world_size) / per_device_batch)` (DistributedSampler pads; `drop_last=False`) and `offset_samples = (global_steps % steps_per_epoch) * per_device_batch * world_size`.
- Produces: `continue_epoch_order(ds, offset, seed, world_size=1, epoch=0)`, which seeds the permutation with `seed + epoch` (DistributedSampler's rule); existing callers are unchanged.
- Semantics of `--resume_data_offset_steps N`: the global optimizer steps already taken, in any epoch. After a resume, the epochs that follow see every document once each, but not in the order an uninterrupted run would have used. This is documented, not fixed.

- [ ] **Step 1: Write the failing tests** in `tests/test_resume_order.py`:

```python
def test_resume_position_epoch_boundary():
    # 1000 docs, 2 ranks x 8 -> ceil(500/8) = 63 steps per epoch
    assert resume_position(63, 1000, 8, 2) == (1, 0)
    assert resume_position(70, 1000, 8, 2) == (1, 7 * 16)

def test_continue_epoch_order_matches_sampler_in_epoch_1():
    full = list(range(N))
    s = DistributedSampler(full, num_replicas=1, rank=0, shuffle=True, seed=SEED); s.set_epoch(1)
    expected = list(s)[40:]
    view = continue_epoch_order(full, 40, SEED, epoch=1)
    s2 = DistributedSampler(view, num_replicas=1, rank=0, shuffle=True, seed=SEED); s2.set_epoch(1)
    assert [view[i] for i in s2][: N - 40] == expected
```

- [ ] **Step 2: Run** `$PY -m pytest tests/test_resume_order.py -v` → the new tests FAIL.
- [ ] **Step 3: Implement both functions, and in `setup` compute `(epoch, offset)` with `resume_position(len(train_ds), ...)` before calling `continue_epoch_order`.** Update the flag's help text to "global optimizer steps already trained (any epoch)".
- [ ] **Step 4: Run** the resume tests → PASS; run the whole `tests/` suite → PASS.
- [ ] **Step 5: Global-step validation.** In `train_model.py`, map `--check_val_every_n_epoch 0` to `None` and pass `val_check_interval` to the Trainer as `int` when it is > 1. Lightning then counts the interval in global steps.
- [ ] **Step 6: Write `tests/test_val_cadence.py`** (marked `@pytest.mark.gpu`, skipped without CUDA). In a `tmp_path` fixture it builds a 700-document `pajama_subset` from the smoke shards (`python -m data.nlp.pajama_subset --num_docs 700`, reusing the download cache) and a 60-index validation file, leaving 640 training documents. It then trains T++ xxs with batch 32 on GPU 0 (20 steps per epoch), `--max_steps 50 --val_check_interval 15 --check_val_every_n_epoch 0`, then reads the CSV log and checkpoint names:

```python
assert val_steps == [14, 29, 44]  # 0-based global steps of the validations, across 2 epoch boundaries
assert ckpt_steps == {15, 30, 45}
```

- [ ] **Step 7: Run it** → PASS. Repeat with a kill at step 25 and `--resume_training_ckpt ... --resume_data_offset_steps 25`: same validation steps, and the first resumed batch equals the sampler's epoch-1 batch 5.
- [ ] **Step 8: Commit.**

### Task 4: Harness run keys

**Files:**
- Modify: `experiments/thinking_study/common/harness.py`, `experiments/thinking_study/common/test_harness.py`

**Interfaces:**
- Produces: `PAPER_CKPTS = REPO / "experiments/scale_to_paper_level/runs/logs/checkpoints"`; `RUN_DIRS["tpp_paper"] = (PAPER_CKPTS, "tpp-paper_20*")`, `RUN_DIRS["s2_paper"] = (PAPER_CKPTS, "s2-paper_20*")`. `run_q2.py` / `run_q3.py` pick these up through `choices=sorted(RUN_DIRS)`.

- [ ] **Step 1: Add a failing test** `test_paper_run_keys_resolve(tmp_path, monkeypatch)` that creates `s2-paper_2026-11-01_00-00-00_/epoch=epoch=1-step=step=1000000-valid_loss=valid_loss=3.5.ckpt` and asserts `ckpt_path("s2_paper", 1000000)` finds it.
- [ ] **Step 2: Run** → FAIL. **Step 3: Add the keys.** **Step 4: Run** the harness tests → PASS (old tests unchanged).
- [ ] **Step 5: Commit.**

### Task 5: Launch scripts, benchmark and smoke runs

**Files:**
- Create: `$WS/run_tpp.sh`, `$WS/run_s2.sh`, `$WS/benchmark.py`

**Interfaces:**
- Consumes: `common.sh` (Task 1).
- Produces: `GPU_IDS=0,1,2,3 bash run_tpp.sh --run_name tpp-paper` and `GPU_IDS=0,1,2,3 bash run_s2.sh --run_name s2-paper`. Per-device batch 32 for both. MCMC flags copied from `../scale_to_16m_sequences/run_s2.sh`.
- Produces: `python benchmark.py {s2,tpp} [extra run-script args]` → `$WS/logs/benchmark_<model>.json` with `steps_per_s` and `eta_h_paper`.

- [ ] **Step 1: Benchmark** with `$WS/benchmark.py` on the smoke build (`--dataset_dir $WS/data/smoke`) on GPUs 0–3 (4 × 32 for both). Record steps/s, peak memory and ETA for 1M steps in `$WS/logs/benchmark_{s2,tpp}.json`.
Expected: S2 ≈ 2.7–3.0 steps/s (ETA ≈ 95–105 h including validation); T++ ≈ 12–17 steps/s (ETA ≈ 18–25 h). If S2 runs out of memory at 32/GPU, use 16 × 2 accumulation and record the change.
- [ ] **Step 2: Smoke runs** of both scripts on the smoke build, 200 steps, `--val_check_interval 100`. Expected: two validations, no NaN, checkpoints at steps 100 and 200.
- [ ] **Step 3: Commit** scripts and benchmark JSONs.

### Task 6: Training

- [ ] **Step 1: Freeze the pre-registration.** Commit this plan with the benchmark numbers filled in, before launching.
- [ ] **Step 2: Launch T++** on GPUs 0–3 in `tmux` session `tpp-paper`, logging to `$WS/logs/tpp_paper.log`. After it crosses the first epoch boundary (≈ step 516k, depending on the built document count), check that validation still lands on multiples of 50k. If it does not, stop and fix before S2.
- [ ] **Step 3: Launch S2** on GPUs 0–3 in `tmux` session `s2-paper` (log `$WS/logs/s2_paper.log`) once T++ has finished and its GPUs are free.
- [ ] **Step 4: Health checks** at 15 min, at 10k steps (end of warmup) and at every validation:
  - no `NaN or Inf gradients detected during MCMC`;
  - validation loss within the 16M run's trajectory at equal sequences seen (S2 4.09 at 3.2M, 3.72 at 16M; T++ 3.83, 3.54), allowing for the different validation set;
  - disk ≥ 500 GB free.
- [ ] **Step 5: On a crash,** resume with `--resume_training_ckpt <abs>/last.ckpt --resume_data_offset_steps <global step>` (Task 3). Record every resume in the log and the report; the replay buffer restarts empty, as in the 16M run.
- [ ] **Step 6: Done when** both runs have the 1M-step checkpoint and 20 validation points.

### Task 7: Evaluation and collection

**Files:**
- Create: `$WS/run_eval.sh`, `$WS/collect_paper.py`, `$WS/test_collect_paper.py`

**Interfaces:**
- Consumes: run keys `s2_paper`, `tpp_paper` (Task 4); `run_q2.py --data`, `run_q3.py --data --skip-part-b`, `analyze_q2.py`, `analyze_q3.py`; `ood_data.OOD_SETS`.
- `run_eval.sh PART=final`, for the Table 3 checkpoint of each model:
  - for `data` in `rp_val` + the 4 OOD sets: Q2 σ = 3 (32 candidates, ≤ 1,024 seqs) and Q3 Part A (k = 1…32, noise 0 and 3);
  - T++ CE on the same tokens (Q2 already records it with `--tpp`).
- `run_eval.sh PART=curve`: Q2 σ = 3 at all 20 S2 checkpoints, on `bigbench_dyck_languages` (C4) and `rp_val` with 1,024 seqs (C5).
- Results go to `$WS/results/{q2,q3}/<key>_step<step>_<data>[_noise<σ>]/`.
- Produces in `collect_paper.py`:
  - `pct_improvement(ce_base: float, ce_variant: float) -> float` (as `collect_ood.py`);
  - `bon_ce(e: np.ndarray, ce: np.ndarray, n: int) -> float` (as `plot_fig6a.py`; arrays are (candidates, tokens));
  - `spearman(x, y) -> float`;
  - `verdicts(results: dict) -> list[dict]`, one row per C1–C7 with value, threshold and verdict;
  - `main()` writes `report/result.md` and `report/figures/{fig6a,fig6b,figB1a,fig7,gap_curve}.png`.

- [ ] **Step 1: Write failing tests** on synthetic arrays:

```python
def test_bon_ce_picks_lowest_energy():
    e = np.array([[1.0, 0.0], [0.0, 1.0]]); ce = np.array([[5.0, 1.0], [2.0, 9.0]])
    assert bon_ce(e, ce, 2) == 1.5

def test_pct_improvement():
    assert pct_improvement(4.0, 4.0) == 0.0 and pct_improvement(4.0, 3.9) > 0

def test_verdict_c2_threshold():
    rows = verdicts(fake_results(mean_ppl_k2=100.0, mean_ppl_k3=89.0))
    assert row(rows, "C2")["verdict"] == "Supported"     # 11% > 10%
    rows = verdicts(fake_results(mean_ppl_k2=100.0, mean_ppl_k3=95.0))
    assert row(rows, "C2")["verdict"] == "Not supported"
```

Add one boundary test per claim C1–C7 in the same style, with the thresholds from §0.2.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Run the eval** on GPUs 0–3 only (`PART=final` on GPUs 0–1, `PART=curve` on GPUs 2–3; ≈ 4–8 h in total, from the OOD run's ≈ 45 min per checkpoint for 4 sets), then `$PY collect_paper.py`. All evaluation, T++ included, runs after S2 finishes, because S2 uses all four GPUs.
- [ ] **Step 6: Commit** scripts, tests and `report/result.md` + figures (results arrays stay ignored).

### Task 8: Report

**Files:**
- Create: `$WS/report/report.md`

- [ ] **Step 1: Write the report** in the format of the scale-16m and OOD reports:
  - 3-bullet TL;DR, setup, data stats (documents built vs 66M, epochs);
  - results per claim with tables;
  - the verdict table C1–C7, interpretation, limitations, reproduction commands.
  - Include the 3.36M → 16M → 128M series for the gap, the Q2 verifier efficiency, and the k = 2 → 3 gain, stating that the 128M point uses a different validation set.
- [ ] **Step 2: Optional, if C3–C6 fail:** run Q2 entropy-vs-energy (`../scale_to_16m_sequences/entropy_vs_energy.py`) and Q3 Part B on the final S2 checkpoint, to see whether the sharpness bias persists at 128M.
- [ ] **Step 3: Commit.**

## Compute and timeline

| Phase | Resources | Wall-clock |
|---|---|---|
| Tasks 1–5 (code, data build, benchmark) | CPU + GPUs 0–3 briefly | ≈ 1 day (build ≈ 12 h runs alongside Tasks 2–5 on smoke data) |
| Task 6: T++ 1M steps | GPUs 0–3 | ≈ 18–25 h (≈ 70–100 GPU-h) |
| Task 6: S2 1M steps, after T++ | GPUs 0–3 | ≈ 95–105 h (≈ 400 GPU-h) |
| Task 7: evaluation | GPUs 0–3 | ≈ 0.5 day |
| **Total** | | **≈ 7 days** |

> [!note] Measured before launch (pre-registration frozen 2026-10-23)
> - Build: **66,387,864 documents** (1.006 × the paper's 66M), 8,400/8,400 shards, 0 missing → 66,354,864 train / 33,000 val; one epoch = 518,398 steps at 4 × 32, so 1M steps ≈ 1.93 epochs.
> - Benchmark on H100 80GB × 4 (GPUs 0–3), 4 × 32: **S2 2.56 steps/s** (≈ 108 h train-only for 1M steps, peak 65.8 GB/GPU, measured while the build used the CPUs); **T++ 38.6 steps/s** (≈ 7.2 h). Expected wall-clock: T++ ≈ 8 h, S2 ≈ 4.7 days including 20 validations.
> - The claims C1–C7 and their thresholds in §0.2 are unchanged.

> [!warning] Limitations known in advance
> - One seed per model; no variance estimate.
> - xxs only. If the paper's thinking model was "Small" (Table D.4), our models are ≈ 8× smaller in non-embedding parameters.
> - The new validation set makes absolute in-distribution losses incomparable with the 3.36M and 16M points; OOD sets and gaps remain comparable.
> - The paper's exact token scoring, Table 3 inference settings (σ, k) and percentage baselines are not stated. §0.1 fixes our choices in advance.
