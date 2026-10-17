---
title: Reproduce EBT xxs scaling (NLP) — Report
date: 2026-10-13
updated: 2026-10-14
tags: [ebt, reproduction, nlp, scaling]
---

# Reproduce EBT xxs scaling (NLP) — Report

> [!summary] TL;DR
> - After 105k steps (≈860M token positions), valid loss is **3.696 (ppl 40.3) for Transformer++**, **3.837 (ppl 46.4) for EBT System 1** and **3.943 (ppl 51.6) for EBT System 2** (§8). Both EBTs are worse in absolute loss, as the paper also reports at its tested scales; the S1 gap shrinks from 0.511 nats at 15k to 0.141 at 105k, but only slowly after 45k.
> - On one H100 NVL, wall-clock is **5.13× (S1) and 5.92× (S2)** that of Transformer++ (9h04m / 10h27m vs 1h46m). All three runs were stable: no NaN, no resume, no spikes beyond batch noise.
> - The MCMC steps matter very differently: S1's second step improves loss by only 0.019 nats, while S2's prediction comes almost entirely from refinement (6.16 after step 1 → 3.94 after step 3, energy drop ≈5). Testing S2's thinking claims (more steps, self-verification) needs inference experiments not run here.

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
   - *Possible effect:* shards are sampled as whole files, so the subset is clustered by snapshot and partition, and slightly less diverse than i.i.d. sampling. Because the run stays under one epoch, there is no repetition effect. Absolute losses may differ from the paper's. Both models use the **same validation split and the same train batch order**. The trainer defaults to the DDP strategy even on one GPU, and `DDPStrategy.setup_distributed()` calls `reset_seed()` (seed 33) after model init but before `random_split` and the shuffling DataLoader run, so neither depends on how much randomness each backbone's init consumed. Empirically, the step-to-step train-loss residuals of the two runs (after removing a 21-point moving average, steps ≥10k) correlate at 0.98, which only happens when both see the same batches.
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

S2 run (§8), from the same shell:

```bash
GPU_ID=0 DATA_ROOT=$EXP/data/smoke $PY $EXP/benchmark.py ebt_s2              # → runs/benchmark_ebt_s2.json (50/350-step timing, batch 32)
GPU_ID=0 bash $EXP/run_ebt_s2_xxs.sh > $EXP/logs/ebt_s2_xxs.log 2>&1      # ~10.5h on one H100 NVL
$PY $EXP/collect_results.py \
  --run "Transformer++=$EXP/runs/logs/csv/baseline-xxs-scaling-105k/version_*/metrics.csv" \
  --run "EBT-S1=$EXP/runs/logs/csv/ebt-xxs-scaling-105k/version_*/metrics.csv" \
  --run "EBT-S2=$EXP/runs/logs/csv/ebt-s2-xxs-scaling-105k/version_*/metrics.csv" \
  --benchmark "Transformer++=$EXP/runs/benchmark_baseline.json" \
  --benchmark "EBT-S1=$EXP/runs/benchmark_ebt.json" \
  --benchmark "EBT-S2=$EXP/runs/benchmark_ebt_s2.json" \
  --out $EXP/report/result.md
```

The committed `benchmark_ebt_s2.json` came from an equivalent one-off script (same 50/350-step protocol at batch 32, logs `logs/bench_s2_b32_{50,350}.log`); that script also showed batch 128 OOMs on the 94 GB GPU (`logs/bench_s2_b128_*.log`).

If a run dies, re-run the same script with `--resume_training_ckpt <absolute path to runs/logs/checkpoints/.../last.ckpt>`. Global step and CSV logging continue in a new `version_<n>` directory, which `collect_results.py` merges.

## 8. EBT System 2 vs System 1 vs Transformer++

A third run trains the same xxs EBT with the paper's **System 2 (S2)** MCMC hyperparameters. It keeps everything else from §2: batch 32, 105k steps, the same data, split and batch order (DDP `reset_seed`, §6.1), and seed 33. It ran on GPU 0 from 2026-10-14 05:20:32 to 15:47:12 UTC, exited 0, with a single `version_0`. It is logged to the same trackio project, `ebt_xxs_scaling`, as run `ebt-s2-xxs-scaling-105k`.

S2-only settings (`run_ebt_s2_xxs.sh`, from paper Table D.4, NLP column):

| Hparam | S1 (§2) | S2 |
|---|---|---|
| MCMC steps (train) | 2 | random 2–3 per batch (`--mcmc_num_steps 1 --randomize_mcmc_num_steps 2 --randomize_mcmc_num_steps_min 2`) |
| MCMC steps (validation) | 2 | 3 (the code uses the maximum when randomness is off) |
| Step size α | 500, learnable (ends at 528.9) | 5, fixed; uniform in [α/2, 2α] per token during training |
| Langevin noise std | 0 | 3.0 on the logits before every step, **also at validation** (`no_langevin_during_eval` is off) |
| Replay buffer | no | yes (size 192; half of each train batch starts from a stored final prediction) |
| `--no_mcmc_detach` / `--truncate_mcmc` | no / no | yes / yes (loss on the final step only) |

### 8.1 Results

From `result.md`. All EBT losses are the final-step loss.

| model | valid loss @105k | valid ppl | Δ vs Transformer++ | wall-clock | × Transformer++ | steps/s (bench) | peak mem (bench) |
|---|---|---|---|---|---|---|---|
| Transformer++ | 3.6964 | 40.30 | — | 1h45m57s | 1.00 | 16.595 | 9,263 MiB |
| EBT-S1 | 3.8373 | 46.40 | +0.141 | 9h03m37s | 5.13 | 3.347 | 38,445 MiB |
| EBT-S2 | 3.9431 | 51.58 | +0.247 | 10h26m40s | 5.92 | 3.016 | 60,999 MiB |

Validation gap by step (nats):

| step | 15k | 30k | 45k | 60k | 75k | 90k | 105k |
|---|---|---|---|---|---|---|---|
| S2 − S1 | −0.096 | +0.146 | +0.143 | +0.144 | +0.121 | +0.113 | +0.106 |
| S2 − Transformer++ | 0.415 | 0.355 | 0.318 | 0.302 | 0.270 | 0.256 | 0.247 |

- S2 is ahead of S1 at 15k, when S1 is still recovering from its large-α warmup (§5.2). From 30k on, S2 is behind S1 and the gap closes slowly, by about 0.04 nats over 75k steps.
- S2 is the worst of the three at 105k: its perplexity is 1.11× S1 and 1.28× Transformer++.
- The S2 validation loss is a noisy estimate: the Langevin noise (std 3) stays on at validation, so a different noise draw gives a slightly different number. Its size was not measured.
- The S2 train loss is not comparable to S1's: half of each train batch starts from replay-buffer predictions rather than noise, and the number of steps is random.

### 8.2 MCMC dynamics: S2 vs S1

`initial loss` is the loss after the first MCMC step; `final-step loss` is after the last one (2 steps for S1, 3 for S2 at validation). `energy gap` is the mean energy after the first step minus after the last.

| valid @ step | S1 initial | S1 final | S1 energy gap | S2 initial | S2 final | S2 energy gap |
|---|---|---|---|---|---|---|
| 15k | 4.795 | 4.804 | −0.005 | 5.988 | 4.708 | 2.86 |
| 30k | 4.267 | 4.255 | 0.014 | 6.214 | 4.401 | 3.87 |
| 60k | 4.015 | 3.999 | 0.016 | 6.147 | 4.143 | 4.66 |
| 90k | 3.883 | 3.865 | 0.024 | 6.308 | 3.978 | 5.13 |
| 105k | 3.856 | 3.837 | 0.029 | 6.164 | 3.943 | 4.97 |

- **S1 is effectively a one-step predictor.** Its first step already gives almost the final quality, and the second step adds 0.019 nats.
- **S2 is a multi-step refiner.** After one step its loss is about 6.2 nats, and it does *not* improve over training (5.99 → 6.16). All of the learning shows up in the later steps: 2.2 nats of improvement and an energy drop of about 5 at 105k.
  - This is what the setup predicts. α=5 is about 100× smaller than S1's learned α, so one step cannot move the noise far. With `--truncate_mcmc`, only the final step's prediction enters the loss.
  - So with S2, the energy landscape is what carries the prediction: the model must be minimized for 3 steps to be useful, and its single-step output is not a usable language model.
- On train data, every one of the 2,100 logged points has final-step loss below initial loss and a positive energy gap. In S1, 366 points had no improvement and 246 had a negative gap (§5.2).
- No NaN or Inf (no `NaN or Inf gradients detected during MCMC` in the log). `pct_gradient_clipped` (share of gradient elements with |g| > 1, §5.3) was 0.0018% at step 49, nonzero at only two later logged points (max 0.00015%, step 199), and 0% from step 200 on. The mean per-tensor gradient norm was 8.0 at step 49 and 0.029–0.49 from step 1,000 on, the same range as S1. S2 had no large early spike like S1's (1,135 at step 49). That is consistent with α=5 being 100× smaller than S1's initial α=500, but we did not test the cause.

### 8.3 What this does and does not test

- The paper uses S2 for its **thinking** results: more MCMC steps at inference and self-verification over several sampled predictions (paper Table 2 reports these as % perplexity improvement on BigBench Dyck Languages). Its regularizers (Langevin noise, random step size and count, replay buffer) are described as making the energy landscape better for thinking, and Table 2 shows that removing Langevin dynamics *improves* single-path "thinking longer" (17.2 vs 7.19). We found no claim in the paper that S2 lowers pretraining loss at a fixed step count.
- So a worse pretraining loss for S2 than for S1 does not contradict the paper. Our *hypothesis*, not tested here, is that part of the S1–S2 gap comes from the noise and randomization that S2 adds on purpose.
- This run does not test the thinking claims. That needs inference with more steps (e.g. 3 → 8) and with N sampled predictions picked by minimum energy, on both checkpoints. These are diagnostics C and D proposed earlier in this project; they have not been run.

### 8.4 Deviations from the paper (S2)

1. **Batch 32 instead of 128.** Paper App. D.1.2 trains the S2 thinking model with batch 128. On one 94 GB H100 NVL, batch 128 runs out of memory (`logs/bench_s2_b128_*.log`: tried to allocate 9.21 GiB with 88.38 GiB in use). Gradient accumulation (32×4) would fit but costs about 39 GPU-h; we kept batch 32 so S2 is comparable step-for-step with S1 and Transformer++.
2. **105k steps instead of 1M.** The paper's thinking model trains for 1M steps. At the measured 3.0 steps/s, 1M steps at batch 32 would take about 92 h (about 368 h at batch 128).
3. **Model size xxs.** The text of App. D.1.2 says the thinking experiments use xxs, but Table D.4 lists model size "Small". We followed the text, which also matches S1.
4. **Data: 3.4M-document RedPajama-V2 subset** instead of the full `sample-100B` (same as §6.1). S2 needs about 3.4M sequences for 105k steps at batch 32, so this run, like S1, stays under one epoch.
5. **Hyperparameters from paper Table D.4, not the repo's current `job_scripts/nlp/pretrain/ebt_s2.sh`.** The script was changed after the paper (commit `19420cb`) to α 0.5, 2–4 steps, `--norm_pred`, `--scale_alpha_with_energy` and FineWeb data with global batch 256. The paper's values were chosen because the goal is to reproduce the paper.
6. **α is not learnable**, following Table D.4 (✗). This is different from S1 (α learnable, LR multiplier 1500).
7. **Validation keeps Langevin noise on and uses the maximum step count (3).** This is the repo's default behaviour, not a choice we made; the paper does not say how its S2 validation loss was computed.
8. **Peak memory and steps/s are benchmark values** (350 steps at batch 32, whole-device `nvidia-smi`), not measurements from the 105k run. The 105k run averaged 2.79 steps/s (105,000 steps in 37,600 s), including validation and data loading; the S1 run averaged 3.22 steps/s.
