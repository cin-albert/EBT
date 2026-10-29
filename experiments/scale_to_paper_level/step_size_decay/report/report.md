---
tags: [ebt, inference, step-size, momentum]
date: 2026-10-29
---

# Step-size decay — does a smaller step explain the EMA momentum gain?

> [!summary] TL;DR
> - **No. Every step-size schedule is worse than the trained constant α = 5, and far worse than EMA momentum.** The arm selected on rp_val (`gd_a3.5`) reaches 84.42 mean OOD ppl at k = 3, against `gd_a5` 82.76 and `ema_m0.5` 72.21. Decaying schedules reach 102.6–384.1 at k = 3 and drift toward a uniform distribution with more steps. D1's pre-registered verdict is **"momentum does more"**, and D2–D5 are not supported.
> - **In the repo's inference loop, α is also the output temperature.** Each update re-anchors at the probabilities, z_{t+1} = p_t − α_t u_t, and p_t ∈ [0, 1]. The logit spread is therefore ≈ α_t times the spread of u_t. Decaying α flattens p_t: entropy climbs to log V, and `gd_a5_g0.5` ends at ppl 50,277 = V, a uniform distribution. A smaller constant α instead makes the 2-cycle worse (64–78% of tokens vs 29.6%).
> - **What EMA does is selective, not global, damping.** Decay arms remove the 2-cycle (0%) but destroy the logit scale. Bias-corrected EMA removes the 2-cycle and keeps the scale along directions where gradients agree. Adding a harmonic decay to EMA (`ema_m0.5_harm`, 424.0) destroys the gain. The reviewer's open question in `../energy_momentum/report/report.md` §2.3 is answered for global step-size schedules. A schedule that shrinks the update without shrinking the logit scale was not tested.

## 1. Setup

| Item | Value |
|---|---|
| Checkpoint | `s2_paper`, step 1M (the momentum report's checkpoint) |
| Sequences | 1,024 per set (rp_val + 4 OOD), batch 16, `torch.manual_seed(seed + b)`, k = 1…32 recorded per step |
| Pairing | Same sequences and seeds as the momentum grid; the constant-schedule path reproduces `gd_a5` bit-exactly (`test_run_decay.py`) |
| Noise | σ ∈ {0, 3}; σ is added to the logits each step and is not decayed |
| Selection | `d*` = lowest rp_val CE at k = 3, σ = 3 among the 7 GD arms |
| Metric | Arithmetic mean of the 4 OOD sets' ppl at k = 3, σ = 3 |

Arms (α_t of update t = 0, 1, …; update 0 leaves the random initial state):

| Arm | α₀, α₁, α₂ | Schedule |
|---|---|---|
| `gd_a2.5`, `gd_a3.5` | 2.5 / 3.5 constant | small constant step |
| `gd_a5_harm` | 5, 2.5, 1.67 | α₀ / (t + 1) |
| `gd_a5_g0.5` | 5, 2.5, 1.25 | α₀ · 0.5^t |
| `gd_a5_g0.7` | 5, 3.5, 2.45 | α₀ · 0.7^t |
| `gd_a7.5_harm` | 7.5, 3.75, 2.5 | α₀ / (t + 1) |
| `gd_a7.5_g0.7` | 7.5, 5.25, 3.67 | α₀ · 0.7^t |
| `ema_m0.5_harm` | 5, 2.5, 1.67 | EMA (μ = 0.5) + harmonic decay |

## 2. Results

### 2.1 Pre-registered claims

| # | Claim | Value | Supported if | Verdict |
|---|---|---|---|---|
| D1 | A step-size schedule explains the EMA gain | `gd_a3.5`: 84.42 (EMA 72.21, `gd_a5` 82.76) | ≤ 73.2 most / ≥ 78.0 momentum does more | **momentum does more** |
| D2 | Decay beats the trained constant step | −2.0% (worse), CI < 0 on 0/4 | > 2%, CI < 0 on ≥ 3/4 | **Not supported** |
| D3 | Decay removes the 2-cycle | `gd_a3.5`: 63.9% (`gd_a5` 29.6%) | < 5% | **Not supported** |
| D4 | Decay stops the drift with k | +30.4% from k = 3 to 32 (EMA +8.5%) | < +2% | **Not supported** |
| D5 | (exploratory) Momentum and decay add up | `ema_m0.5_harm` 424.0 vs 72.21 | CI < 0 on ≥ 3/4 | **Not supported** |

Note on D3 and D4: the selection picked a *constant* small α, because every decaying arm is worse on rp_val (CE 3.79–5.01 vs 3.59). The decaying arms do pass D3 (0% 2-cycles, §2.3), but they fail on perplexity, so the claim the plan meant to test still fails.

### 2.2 Perplexity (k = 3, σ = 3)

| Set | T++ ours | `gd_a5` | `ema_m0.5` | `gd_a3.5` (d*) | `ema_m0.5_harm` | d* vs `gd_a5`: ΔCE [95% CI] | d* vs EMA: ΔCE [95% CI] |
|---|---|---|---|---|---|---|---|
| Pretrain (val) | 30.36 | 34.63 | 33.73 | 36.23 | 156.35 | +0.045 [+0.043, +0.047] | +0.071 [+0.069, +0.074] |
| GSM8K | 40.38 | 44.55 | 42.15 | 46.52 | 242.80 | +0.043 [+0.041, +0.045] | +0.099 [+0.095, +0.102] |
| SQuAD | 50.03 | 56.41 | 54.00 | 58.44 | 306.69 | +0.035 [+0.033, +0.037] | +0.079 [+0.076, +0.082] |
| BB Math QA | 80.34 | 88.32 | 70.87 | 90.03 | 492.19 | +0.019 [+0.016, +0.022] | +0.239 [+0.234, +0.245] |
| BB Dyck | 107.96 | 141.77 | 121.82 | 142.70 | 654.30 | +0.006 [−0.005, +0.018] | +0.158 [+0.144, +0.172] |
| Mean of 4 OOD | 69.68 | 82.76 | 72.21 | 84.42 | 424.00 | | |

Mean OOD ppl at k = 3, σ = 3, of the other arms (all post hoc):
- **Small constant α:** `gd_a2.5` 90.0.
- **Decaying schedules:** `gd_a7.5_g0.7` 102.6, `gd_a5_g0.7` 120.9, `gd_a7.5_harm` 126.3, `gd_a5_harm` 171.4, `gd_a5_g0.5` 384.1.

Over all k ≤ 32, the best new arm is `gd_a3.5` at k = 3: 84.4 at σ = 3 (EMA 72.2) and 79.4 at σ = 0 (EMA 72.7); see `report/result.md`.

![Schedules, OOD perplexity vs k, and output entropy](figures/ce_vs_k.png)

### 2.3 Mechanism: α sets the logit scale

The loop re-anchors at the probabilities each step (`../energy_momentum/momentum.py`):

```python
p = softmax(z + noise)            # p in [0, 1]
z = p - alpha_t * u               # logits = O(1) + alpha_t * u
```

Because p_t is bounded, the logits' spread is set almost entirely by α_t · u_t. α acts as an inverse temperature as much as a step size. The data agree:

| Arm (rp_val, σ = 0) | Entropy of p_t, t = 1…6 (nats; log V = 10.83) | 2-cycle share, t = 31 |
|---|---|---|
| `gd_a5` | 0.67, 4.88, 4.27, 4.77, 4.31, 4.75 | 29.6% |
| `ema_m0.5` | 0.67, 1.76, 3.14, 4.05, 4.43, 4.53 | 0.0% |
| `gd_a3.5` | 1.12, 6.47, 4.35, 6.34, 4.41, 6.30 | 63.9% |
| `gd_a5_harm` | 0.67, 7.94, 6.91, 8.96, 8.49, 9.31 | 0.0% |
| `gd_a5_g0.5` | 0.67, 7.94, 8.35, 9.94, 10.44, 10.71 | 0.0% |
| `gd_a7.5_g0.7` | 0.40, 4.69, 5.87, 6.72, 7.74, 8.44 | 0.0% |

- **Decay removes the oscillation by flattening the output.** Every decaying arm has 0% 2-cycles, but its entropy rises toward log V. At k = 32, `gd_a5_g0.5` has mean OOD ppl 50,277, which is exactly the uniform distribution over the vocabulary. Removing the 2-cycle is therefore not sufficient for the EMA gain.
- **A smaller constant α makes the oscillation worse.** The 2-cycle share is 63.9% at α = 3.5 and 78.1% at α = 2.5, against 29.6% at α = 5. The entropy alternates between ≈ 4.4 and ≈ 6.4 nats, so perplexity is good at odd k and bad at even k. For `gd_a2.5` at σ = 0, ppl is 83.2 at k = 3 and 443.9 at k = 2. The best odd-k values (83.2 / 79.4 at k = 3, σ = 0) are near `gd_a5` (77.8) and still above EMA (72.7).
- **EMA damps only the alternating component.** Bias-corrected EMA divides the weighted gradient sum by the weights' sum. Where consecutive gradients agree, |u| ≈ |g| and the logit scale is kept. Where they alternate, they cancel. Adding a harmonic decay on top (`ema_m0.5_harm`) shrinks the scale again and ppl explodes (424.0 at k = 3). In this loop, EMA's benefit cannot be reproduced by any global scalar on the step.

## 3. Interpretation

- **The momentum report's main caveat is answered for global schedules.** "The trained α = 5 is too large at inference, and a smaller or decaying α would do as well" is ruled out: every constant α < 5 and every decay schedule tested is worse than α = 5, and far worse than EMA.
- **"Step size" is the wrong control in a re-anchored loop.** The plan treated α as a pure step size. In this loop it also sets the temperature, so shrinking α trades the 2-cycle for a flat distribution. EMA wins because it is direction-selective. The plan's prediction (decay recovers 73–78) was wrong.
- **What remains untested** is a control that damps the update without changing the logit scale. Examples are a non-re-anchored loop z_{t+1} = z_t − α_t g_t, or decaying α while rescaling the logits to a fixed norm. Such a control would separate "averaging directions" from "selective damping of alternating components". These are new update rules rather than step-size controls, and for this question they are close to what EMA already is.

> [!warning] Limitations
> - One checkpoint (`s2_paper` 1M), one noise seed set, and 1,024 sequences per set. The bootstrap CI resamples sequences only.
> - The noise σ is not decayed with α. At σ = 0, which has no noise, the decay arms fail the same way (§2.3), so noise is not the cause.
> - The schedules are coarse (harmonic, geometric 0.5 / 0.7, constant 2.5 / 3.5). None of them decays only after the first 2–3 updates, which is where EMA's own k = 3 optimum sits. Given the entropy mechanism, a late decay would also flatten p at large k.
> - The full-set Table 3 protocol (plan Task 5) was not run, because D1's verdict was not "explains most" or "partial".

## 4. Reproduction

```bash
cd code/EBT/experiments/scale_to_paper_level/step_size_decay
PY=/home/quangnx/miniconda3/envs/ebt312/bin/python
$PY -m pytest -q                                          # incl. GPU test: const schedule == gd_a5 grid
GPU_ID=0 ARMS="gd_a5_harm gd_a2.5" bash run_all.sh        # ~20 min per GPU
GPU_ID=1 ARMS="gd_a5_g0.5 gd_a3.5" bash run_all.sh
GPU_ID=2 ARMS="gd_a5_g0.7 gd_a7.5_g0.7" bash run_all.sh
GPU_ID=3 ARMS="gd_a7.5_harm ema_m0.5_harm" bash run_all.sh
export PYTHONPATH=$PWD/../../..:$PWD/../../thinking_study/common:$PWD/../../thinking_study/q3_energy_landscape:$PWD/..:$PWD/../energy_momentum:$PWD
$PY collect_decay.py                                      # report/result.md, report/figures/ce_vs_k.png
```
