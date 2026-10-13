---
title: Reproduce EBT xxs scaling (NLP) — Report
date: 2026-10-13
tags: [ebt, reproduction, nlp, scaling]
---

# Reproduce EBT xxs scaling (NLP) — Report

> [!summary] TL;DR
> - After 105k steps (≈860M token positions), **Transformer++ reaches valid loss 3.696 (ppl 40.3)** and **EBT reaches 3.837 (ppl 46.4)**. EBT's perplexity is about 15% worse at xxs scale. The paper also reports worse absolute loss for EBT at its tested scales.
> - EBT's validation gap to Transformer++ shrinks from 0.511 nats at 15k steps to 0.141 nats at 105k steps. Over the last 45k steps it shrinks only slowly (0.158 → 0.141).
> - Measured on one H100 NVL, **EBT cost 5.13× the wall-clock** of Transformer++ (9h04m vs 1h46m) and 4.15× the peak memory (37.5 GiB vs 9.0 GiB). The run was stable: no NaN, no resume, and no loss spikes beyond the batch noise Transformer++ also shows.

## 1. Summary

The goal was to reproduce the smallest point (xxs, about 6.2M non-embedding parameters) of the paper's NLP pretraining scaling comparison between EBT (System 1, 2 MCMC steps, `time_embed`) and Transformer++, on a single GPU. Both models were trained for 105k steps with the paper's hyperparameters, on a 3.4M-document shard sample of RedPajama-V2 `sample-100B`.

At 105k steps, EBT's validation loss is 0.141 nats higher than Transformer++. In perplexity that is 46.4 vs 40.3, a ratio of 1.15×. EBT took 5.13× the wall-clock time.

The direction agrees with the paper. The paper finds that EBT has worse absolute loss at small scale and claims a higher *scaling rate*. A single model size cannot test the scaling-rate claim.

## 2. Setup

| Item | Value |
|---|---|
| GPU | 1× NVIDIA H100 NVL (physical index 0, `CUDA_VISIBLE_DEVICES=0`) for both runs |
| Env | `ebt312` (`/mnt/data/conda-envs/quangnx/ebt312`), pinned in `ebt312-constraints.txt`: torch 2.4.0, pytorch-lightning 2.4.0, transformers 4.44.2, tokenizers 0.19.1, datasets 3.0.0, huggingface-hub 0.36.2, trackio 0.20.2 |
| Code | branch `repro/xxs-scaling`, commit `d388ac0` (runs were launched from this tree) |
| Precision | `--float_precision 32-true`, `--set_matmul_precision medium` |

Shared hyperparameters (paper Table D.1–D.3, App. D.1.1):

| Hparam | Value |
|---|---|
| model_size | `xxs` (6 layers, D=384, 6 heads) |
| `--ffn_dim_multiplier` | `1` |
| Params | 44.8M total, of which non-embedding: EBT 6,199,296 and Transformer++ 6,198,144 (paper: 6.18M) |
| `--context_length` | `256` |
| Batch | `32 × 1 accum × 1 GPU` (paper App. D.1.1: xxs uses batch 32 for the parameter/FLOP scaling runs; `ebt_s1.sh` uses a global batch of 256 for other experiments) |
| `--peak_learning_rate` | `0.0012` |
| `--max_steps` / `--max_scheduling_steps` | `105000` / `105000` |
| `--warm_up_steps` / `--min_lr_scale` | `10000` / `10` |
| `--weight_decay` / `--gradient_clip_val` | `0.01` / `1.0` |
| Tokenizer | `EleutherAI/gpt-neox-20b` |
| Seed | 33 |
| Validation | `--validation_split_pct 0.0098` (33,320 docs), every 15,000 steps |

EBT-only settings: `--normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise --mcmc_step_size_learnable --mcmc_step_size 500 --mcmc_step_size_lr_multiplier 1500 --mcmc_num_steps 2`.

Run commands and where their outputs went:

```bash
EXP=experiments/reproduce_xxs_scaling
GPU_ID=0 bash $EXP/run_baseline_xxs.sh > $EXP/logs/baseline_xxs.log 2>&1
GPU_ID=0 bash $EXP/run_ebt_xxs.sh      > $EXP/logs/ebt_xxs.log 2>&1
```

- CSV metrics: `runs/logs/csv/{baseline,ebt}-xxs-scaling-105k/version_0/metrics.csv`
- Checkpoints: `runs/logs/checkpoints/<run_name>_<timestamp>_/`
- trackio: `runs/trackio/`, project `ebt_xxs_scaling`
- Benchmarks: `runs/benchmark_{baseline,ebt}.json`

## 3. Data

Source: `logs/data_stats.json`, built with `python -m data.nlp.pajama_subset` (seed 33).

| Stat | Value |
|---|---|
| Documents | 3,400,000 (train 3,366,680 / valid 33,320) |
| Shards used / missing | 383 / 0, out of 8,400 `sample-100B` tags |
| Language mix | en 2,717,441 (79.9%) · fr 217,251 · de 207,026 · es 170,670 · it 87,612 |
| Short docs skipped (<2 tokens) | 2 |
| Mean length (truncated to 257 tokens) | 226.6 tokens; 77.1% of documents fill all 257 |
| Token positions seen in 105k steps | 105,000 × 32 × 256 = 860,160,000 (upper bound, since short docs are padded) |
| Epochs | 0.998 (one epoch = 105,209 steps), so no document was repeated |
| Disk | raw cache 8.4 GB, tokenized memmap 1.7 GB |

## 4. Results

From `result.md`. The EBT loss is `valid_final_step_loss`, the loss of the prediction after the last MCMC step. Perplexity here is `exp(loss)`. The trainer also logs `valid_perplexity` as a per-batch mean, which is slightly higher: 40.52 for Transformer++ and 46.65 for EBT at 105k.

### Final validation and throughput

| model | step | valid loss | valid ppl | steps/s | tokens/s | peak mem (MiB) | ETA 105k (h) |
|---|---|---|---|---|---|---|---|
| Transformer++ | 104999 | 3.6964 | 40.30 | 16.595 | 135949 | 9263 | 1.76 |
| EBT | 104999 | 3.8373 | 46.40 | 3.347 | 27417 | 38445 | 8.71 |

### Validation curve

| step | Transformer++ loss | Transformer++ ppl | EBT loss | EBT ppl |
|---|---|---|---|---|
| 14999 | 4.2930 | 73.19 | 4.8040 | 122.00 |
| 29999 | 4.0458 | 57.16 | 4.2550 | 70.46 |
| 44999 | 3.9273 | 50.77 | 4.1022 | 60.47 |
| 59999 | 3.8406 | 46.56 | 3.9987 | 54.53 |
| 74999 | 3.7727 | 43.50 | 3.9218 | 50.49 |
| 89999 | 3.7220 | 41.35 | 3.8650 | 47.70 |
| 104999 | 3.6964 | 40.30 | 3.8373 | 46.40 |

## 5. Analysis

### 5.1 Quality

Gap in validation loss, EBT minus Transformer++ (nats):

| step | 15k | 30k | 45k | 60k | 75k | 90k | 105k |
|---|---|---|---|---|---|---|---|
| Δ loss | 0.511 | 0.209 | 0.175 | 0.158 | 0.149 | 0.143 | 0.141 |

- EBT starts much worse because of its warmup phase (see §5.2), then catches up quickly by 30k steps.
- After 30k steps the gap keeps shrinking, but slowly: by about 0.07 nats over the last 75k steps. A straight-line extrapolation would not close the gap within a reasonable budget at this size. This curve is a single run with one learning-rate schedule, so it says nothing about the scaling rate across model sizes.
- Final train loss is 3.855 for EBT (final MCMC step; 3.867 averaged over steps) and 3.721 for Transformer++. These are single-batch values at step 104,999. Both are close to the validation loss, so neither model overfits (training was under one epoch).

### 5.2 EBT dynamics

**α, the learnable MCMC step size** (initialized at 500, LR multiplier 1500):

| step | 49 | 5k | 9.3k (min) | 20k | 40k | 60k | 80k | 105k |
|---|---|---|---|---|---|---|---|---|
| α | 499.8 | 430.6 | 418.3 | 428.6 | 462.3 | 497.8 | 521.8 | 528.9 |

- α falls during the 10k-step LR warmup and bottoms out at step 9,299, near the end of warmup.
- After that it rises almost monotonically, ending above its initial value. Only 25 logged points after the minimum show a drop of more than 0.5.
- It flattens out as the cosine LR decays: +7 between 80k and 105k.
- So the model ends up preferring a *larger* gradient step on the energy landscape than the initial value.

**Does the second MCMC step help?** We compare the loss after the first step (`initial_loss`) with the loss after the final step (`final_step_loss`). We also look at the energy drop between them (`initial_final_pred_energies_gap`).

| split / step | initial loss | final-step loss | improvement | energy gap |
|---|---|---|---|---|
| train @1k | 7.513 | 7.548 | −0.035 | 0.010 |
| train @5k | 5.552 | 5.589 | −0.037 | −0.027 |
| valid @15k | 4.795 | 4.804 | −0.009 | −0.005 |
| valid @30k | 4.267 | 4.255 | +0.012 | 0.014 |
| valid @60k | 4.015 | 3.999 | +0.017 | 0.016 |
| valid @90k | 3.883 | 3.865 | +0.018 | 0.024 |
| valid @105k | 3.856 | 3.837 | +0.019 | 0.029 |

- **Early (≤ about 15k steps):** the second step makes predictions *worse*, and energy often does not decrease. This matches what the 15-minute health check saw.
- **Later:** the step reliably lowers both energy and loss, and the benefit keeps growing.
- Over training, 366 of 2,100 logged train points have a final-step loss that is no better than the initial loss. Only 12 of these come after 30k steps, and the last is at step 71,699. The energy gap is negative at 246 logged points, the last at step 69,699.
- The benefit stays small: 0.019 nats ≈ 1.9% perplexity at 105k. Most of EBT's prediction quality comes from the first optimization step.
- `valid_loss`, the trainer's average over MCMC steps, is 3.847. The final-step value used in §4 is 3.837.

### 5.3 Stability

- **No NaN or Inf.** The log contains no `NaN or Inf gradients detected during MCMC` error. There was no resume: the run has a single `version_0`.
- **Large gradients appear only at the very start of EBT.** The trainer's `pct_gradient_clipped` metric is the share of individual gradient *elements* with |g| > 1.0 before clipping, sampled every 50 steps; it is not a count of norm-clipping events, and the true total gradient norm was not logged. At step 49 this share was 10.7% and the mean per-tensor gradient norm was 1135, while the first-step loss was 4.9e3 (`train_loss`, averaged over MCMC steps, was 3.5e3) because the random-noise initial condition is scaled by α=500 in an untrained model. From step 1,000 on, the share is 0% and the mean per-tensor norm stays in 0.025–0.52. For Transformer++ the share is 0% throughout.
- **Batch-level noise is the same for both models.** After 5k steps, the train loss jumps by more than 0.3 between consecutive logged points 53 times for EBT and 50 times for Transformer++, out of 2,000 points each. These are batch noise, not instability.

### 5.4 Cost

| | Transformer++ | EBT | EBT / Transformer++ |
|---|---|---|---|
| Wall-clock, 105k steps (start → end files) | 1h45m57s | 9h03m37s | 5.13× |
| Steady-state steps/s (benchmark) | 16.595 | 3.347 | 4.96× |
| Peak GPU memory | 9,263 MiB | 38,445 MiB | 4.15× |

- The paper's cost ratio for 2-step S1 EBT is about 6.66× *training FLOPs*. The measured wall-clock ratio is lower, at 5.1×.
- We did not profile where the difference comes from. Fixed per-step costs, such as data loading, the optimizer, and the embedding and output head shared by both models, would dilute the ratio. This is a hypothesis, not a measurement.
- The higher memory comes from second-order backprop through the MCMC steps (`create_graph=True`).
- Peak memory is whole-device `nvidia-smi` usage during the 350-step benchmark on the smoke subset, not a measurement from the 105k runs. Both fit easily on one 94 GB H100 NVL at batch size 32.

### 5.5 Comparison with the paper

- The paper reports that EBT has worse absolute loss than Transformer++ at all tested scales, and claims up to about 35% higher *scaling rate* (see the [[Energy-Based Transformers are Scalable Learners and Thinkers|paper summary]]). Our result agrees on the first point: EBT perplexity is 1.15× Transformer++ at xxs.
- The paper does not state xxs perplexities in the text. We did not read values off its scaling figures, so **no quantitative match to the paper's xxs point is claimed**.
- Testing the scaling-rate claim needs at least two more model sizes (xs, s) trained with the same protocol.

## 6. Deviations and limitations

1. **Data:** the paper trains on all of `sample-100B`, about 66M documents (random split 66M / 33K). We sampled shards with seed 33 from the same 8,400 tags until we had **3.4M documents**, about 5% of the corpus, then random-split about 33K for validation. Each document keeps its first 257 tokens, as in `pajama_dataloader.py`.
   - *Possible effect:* shards are sampled as whole files, so the subset is clustered by snapshot and partition, and slightly less diverse than i.i.d. sampling. Because the run stays under one epoch, there is no repetition effect. Absolute losses may differ from the paper's. Both models draw from the same 3.4M-document subset, but **not** the same validation split or train order: `random_split` and the shuffling DataLoader use the global RNG after model init, which consumes different amounts of randomness for each backbone (the reviewer measured only 329 of 33,320 validation docs shared). The sampling noise of a 33K-document mean is about 0.005 nats, far below the 0.141-nat gap, so the conclusion is unlikely to change.
2. **`max_scheduling_steps = 105000`:** the paper does not give this value. Cosine decay finishes exactly at the end of training. A different schedule changes the absolute losses for both models.
3. **`ebt_type time_embed`:** the paper does not state this for the scaling runs. We took it from `ebt_s1.sh` ("System 1 EBT used for many experiments in the paper"). Other EBT backbones (`default`, `adaln`) might give a different gap.
4. **Logger:** trackio plus CSV instead of wandb. This has no effect on training.
5. **Single seed (33), single model size, single GPU type.** There are no error bars. The 0.141-nat gap is far larger than the batch noise seen in training, but the difference between seeds is unmeasured.
6. **Hardware:** H100 NVL instead of the paper's A100. Only wall-clock is affected, not the per-step loss.

## 7. Reproduction

Run these from the repo root, in order. Logs go to `experiments/reproduce_xxs_scaling/logs/`.

```bash
EXP=$PWD/experiments/reproduce_xxs_scaling
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface

# 0. env + tests
$PY -m pip install -c $EXP/ebt312-constraints.txt "huggingface-hub==0.36.2" "trackio==0.20.2" "pytest==8.3.3"
$PY -m pytest tests/ -q

# 1. data (smoke 20K docs, full 3.4M docs; ~42 min for full)           → logs/build_{smoke,full}.log
$PY -m data.nlp.pajama_subset --dataset_dir $EXP/data/smoke --num_docs 20000 --download_workers 8 > $EXP/logs/build_smoke.log 2>&1
$PY -m data.nlp.pajama_subset --dataset_dir $EXP/data/full  --num_docs 3400000 --download_workers 16 > $EXP/logs/build_full.log 2>&1

# 2. smoke runs (30 steps each, plus resume check)                      → logs/smoke_*.log
DATA_ROOT=$EXP/data/smoke GPU_ID=0 bash $EXP/run_baseline_xxs.sh --run_name smoke-baseline --wandb_project ebt_xxs_smoke --max_steps 30 --val_check_interval 10 --limit_val_batches 5

# 3. benchmark                                                          → logs/bench_*.log, runs/benchmark_*.json
GPU_ID=0 DATA_ROOT=$EXP/data/smoke $PY $EXP/benchmark.py baseline
GPU_ID=0 DATA_ROOT=$EXP/data/smoke $PY $EXP/benchmark.py ebt

# 4. full runs (~1.8h and ~9.1h on one H100 NVL)                        → logs/{baseline,ebt}_xxs.log
GPU_ID=0 bash $EXP/run_baseline_xxs.sh > $EXP/logs/baseline_xxs.log 2>&1
GPU_ID=0 bash $EXP/run_ebt_xxs.sh      > $EXP/logs/ebt_xxs.log 2>&1

# 5. tables                                                             → report/result.md
$PY $EXP/collect_results.py \
  --run "Transformer++=$EXP/runs/logs/csv/baseline-xxs-scaling-105k/version_*/metrics.csv" \
  --run "EBT=$EXP/runs/logs/csv/ebt-xxs-scaling-105k/version_*/metrics.csv" \
  --benchmark "Transformer++=$EXP/runs/benchmark_baseline.json" \
  --benchmark "EBT=$EXP/runs/benchmark_ebt.json" \
  --out $EXP/report/result.md
```

If a run dies, re-run the same script with `--resume_training_ckpt <absolute path to runs/logs/checkpoints/.../last.ckpt>`. Global step and CSV logging continue in a new `version_<n>` directory, which `collect_results.py` merges.
