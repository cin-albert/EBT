---
title: "Q3 — Energy landscape and update-map properties"
date: 2026-10-15
tags: [ebt, thinking-study, energy-landscape]
---

# Q3 — Energy landscape and update-map properties

> [!summary] TL;DR
> - EBT inference is a **fixed-point iteration** p ← softmax(p − α∇_pE(p)), whose fixed points are the stationary points of Φ(p) = α·E(p) − H(p) − ½‖p‖², not plain GD on E. The measurements fit this: along the update direction, the step α is exactly the line-minimizer of Φ (median s* = 1.0 for both models and all states), while E keeps decreasing past it.
> - The loop **converges fast and contracts locally** (spectral radius of the update Jacobian at step 32: median 0.49 for S2, 0.35 for S1; >99% of tokens below 1), but **extra steps do not help loss**: the best CE is at the trained step count (S2: 3.943 at k=3 → 4.041 at k=32; S1: 3.841 at k=2 → 3.871). E itself is **non-convex** in the predicted embedding: 0% of tokens are locally convex, and 33–56% of the Hessian eigenvalues are negative.
> - Implications for a training-time optimizer swap: updates live in a **~20–30-dimensional effective subspace** (participation ratio of ∇_eE out of D = 384), gradients across steps are almost collinear (cos ≈ 0.998–1.0), and the curvature is very anisotropic (λ_max/median|λ| ≈ 13–75). These point to preconditioning or low-dimensional search as plausible levers, and away from momentum. Energy decrease is only weakly aligned with loss decrease (Spearman 0.4 for S2, ≈0 for S1 after step 2).

## 1. Setup

- Checkpoints: S2 and S1 at step 105k (see [[THINKING_STUDY_PLAN]] §2). Correctness gate passed (`../common/results/gate.json`).
- **Part A:** 256 validation sequences (55,825 target tokens), 32 MCMC steps, Langevin noise 0 for both models, plus noise 3 for S2 (its training value). One initial-noise draw per token.
- **Part B:** 32 sequences (6,949 tokens), noise 0, at three states: the initial noise (t = 0), the last trained step (S2 t = 2, S1 t = 1), and t = 31.
  - Exact per-token Hessian of E with respect to the predicted embedding e = W_v p (384 × 384, via 384 Hessian–vector products; tokens are independent, gate check 3).
  - Spectral radius ρ of the Jacobian of the update map T, by 30-step power iteration (Gelfand estimate).
  - Line profile q(s) = softmax(p − s·α·∇_pE) for s ∈ [0, 2], where s = 1 is the actual update.
- Code: `run_q3.py`, `analyze_q3.py`. Raw arrays: `results/<model>_step105000/*.npz` (not in git); metrics: `results/*/metrics.json`.

## 2. Mechanism: what the loop optimizes

From `model/nlp/ebt.py:115-160`, the gradient is taken with respect to the probabilities p = softmax(z), and the state is overwritten with z_{t+1} = p_t − α∇_pE(p_t). Setting the Lagrangian gradient of Φ on the simplex to zero gives α∇_pE + log p + 1 − p = const, i.e. p = softmax(p − α∇_pE(p)). So:

- the fixed points of the loop are the stationary points of Φ, an energy with an entropy regularizer at temperature 1/α (S1: α = 528.9, almost pure energy; S2: α = 5, strong entropy term);
- the previous state carries over only through p ∈ [0, 1], so the loop has little memory, and every update lies in the row space of W_v (≤ 384 dims).

Empirical check: along the actual update direction, the median line-minimizer of Φ is **s = 1.0 at every measured state of both models**, while the minimizer of E is s = 2.0 (the edge of the scanned range; E is still decreasing). The update is a line-optimal step on Φ, not on E.

## 3. Results

### 3.1 Thinking longer (Part A, noise 0)

Mean CE of the prediction after k steps:

| k | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| S2 (α = 5) | 6.282 | 4.143 | **3.943** | 4.078 | 3.972 | 4.060 | 4.053 | 4.044 | 4.041 |
| S2, noise 3 | 6.179 | 4.125 | **3.955** | 4.026 | 3.973 | 4.000 | 3.994 | 3.989 | 3.992 |
| S1 (α = 528.9) | 3.861 | **3.841** | 3.846 | 3.852 | 3.857 | 3.860 | 3.864 | 3.869 | 3.871 |

- Both models are best at exactly the step count they were trained and validated with. Beyond it, CE rises and plateaus: +0.098 nats for S2 and +0.030 for S1 at k = 32.
- S2 oscillates for a few steps (k = 3, 4, 5, 6: 3.943, 4.078, 3.972, 4.060), and so does the mean entropy of p_t (1.23 → 5.59 → 4.62 → 5.29 at t = 1–4). This is consistent with ρ > 1 at the start (§3.3).
- S2 with noise 3 has no fixed point: the change between consecutive predictions stays at TV ≈ 0.89 per step, and only 4.6% of tokens settle within 32 steps.

### 3.2 Convergence and per-token adaptivity

| | S2, noise 0 | S2, noise 3 | S1, noise 0 |
|---|---|---|---|
| Median steps until TV(p_{t+1}, p_t) < 0.01 | 7 | 32 (not converged) | 3 |
| Converged by the trained step count | 6.4% | 1.4% | 3.5% |
| Converged within 32 steps | 82.6% | 4.6% | 96.9% |
| Spearman(steps to converge, entropy) | 0.81 | 0.33 | 0.18 |
| Spearman(steps to converge, CE) | 0.51 | 0.35 | 0.20 |
| Spearman(energy, CE) across tokens, at trained steps | 0.23 | 0.12 | 0.36 |

- **Facet 1 (adaptive compute) holds for S2 without noise:** high-entropy, hard tokens take longer to converge (ρ_s = 0.81 with entropy). It is much weaker for S1.
- Nearly no token has converged at the trained step count, yet stopping there gives the lowest loss. Convergence of the loop and quality of the prediction are different targets.

### 3.3 Local geometry (Part B)

| | S2 t=0 | S2 t=2 (trained) | S2 t=31 | S1 t=0 | S1 t=1 (trained) | S1 t=31 |
|---|---|---|---|---|---|---|
| Median λ_min of ∇²_eE | −7.37 | −1.36 | −1.25 | −0.038 | −0.045 | −0.045 |
| Median λ_max | 4.40 | 4.79 | 3.38 | 0.044 | 0.051 | 0.050 |
| Share of negative eigenvalues | 91% | 37% | 33% | 59% | 56% | 56% |
| Tokens locally convex | 0% | 0% | 0% | 0% | 0% | 0% |
| λ_max / median \|λ\| | 13 | 75 | 56 | 17 | 21 | 21 |
| Median ρ(Jacobian of T) | 1.39 | 0.67 | 0.49 | 0.42 | 0.35 | 0.35 |
| Share contracting (ρ < 1) | 39% | 79% | 99.7% | 88% | 97% | 99.4% |
| Participation ratio of ∇_eE (of 384) | 29.8 | 20.1 | 20.5 | 30.2 | 28.7 | 28.6 |
| Share of ‖g_p‖² in top 1% of vocab | 10% | 9% | 9% | 18% | 20% | 20% |
| Kurtosis of g_p (Gaussian = 3) | 4.5 | 3.9 | 3.9 | 8.6 | 10.7 | 11.0 |
| Line argmin s: CE / E / Φ | 0.5 / 2.0 / 0.375 | 1.6 / 2.0 / **1.0** | 1.75 / 2.0 / **1.0** | 1.25 / 2.0 / **1.0** | 1.25 / 2.0 / **1.0** | 1.25 / 2.0 / **1.0** |

- **E is a saddle-rich, non-convex function** of the predicted embedding: there are no locally convex tokens at any state. The loop still contracts because the entropy term in Φ is strongly convex. E's Hessian is not the Hessian of the map, and ρ < 1 is what matters for convergence.
- **The curvature is very anisotropic:** a few large eigenvalues against a near-zero bulk.
- **The effective dimension is small:** the per-token energy gradients use about 20 (S2) to 29 (S1) of the 384 available dimensions.
- **Energy and loss disagree on the step size:** along the update, CE is minimized at s ≈ 1.25 (S1) and 1.6–1.75 (S2), E keeps falling to s = 2, and Φ is minimized at s = 1.
- **Alignment along the trajectory:**
  - In 17% (S2) and 25% (S1) of the steps, energy decreases while CE increases.
  - Per step, Spearman(ΔE, ΔCE) is about 0.4 for S2, but falls to ≈0 for S1 after step 2 (0.55 → 0.14 → 0.07 → 0.0).
  - Once S1 is past its trained steps, lowering energy carries no information about the loss.
- **Successive gradients are almost collinear:** cos(g_t, g_{t−1}) ≈ 0.97 at step 1 and 0.998–1.0 afterwards, for both models.

## 4. Implications for replacing plain GD in training (Gap 3)

These are hypotheses for a training-time study, ranked by how directly the data supports them. They are not results.

| Lever | Evidence here | Expected effect |
|---|---|---|
| **Low-dimensional search** (gradient-free or second-order in a ~20–30-dim subspace of the 384-dim row space of W_v) | Participation ratio 20–30; every update is in W_v's row space by construction | Makes CMA-ES or exact Newton steps per token affordable. Needs a non-unrolled training signal for gradient-free methods (§5 of the plan) |
| **Preconditioning** (Adam-like per-direction scaling, or a learned or Newton preconditioner) | λ_max/median\|λ\| ≈ 20–75; a few stiff directions | Could equalize progress across directions. Note the stiff directions are where E is convex; the bulk is flat or negative |
| **Step-size / temperature control** | Φ is line-minimized at s = 1 while CE prefers s ≈ 1.25–1.75; the trained step count beats more steps | The effective objective is Φ at temperature 1/α. Learning α per step or per token (S1 already learns a global α) is a cheaper lever than a new optimizer |
| **Momentum / Nesterov** | Gradients already collinear (cos ≈ 1) and the map contracts (ρ ≈ 0.35–0.5) | Little to gain for loss. It may speed fixed-point convergence, but convergence is not what limits loss here |
| **Stopping rule** | Steps to converge correlate with entropy (0.81 for S2) | Per-token adaptive compute is feasible for S2. It should stop near the trained step count, not at convergence |

> [!warning] Scope
> - Single checkpoint per model (105k), xxs, one seed. Part B uses 6,949 tokens.
> - These landscapes were co-trained with this specific update rule. A model trained with a different inner optimizer would learn a different landscape, so §4 states plausibility and mechanism only.
> - ρ is a 30-step power-iteration estimate. It can be inaccurate when the dominant eigenvalues are complex or nearly tied.

## 5. Reproduction

```bash
cd refs/EBT/experiments/thinking_study
GPU_ID=4 bash q3_energy_landscape/run_all.sh          # ~25 min on one H100 NVL
PYTHONPATH=$PWD/common /mnt/data/conda-envs/quangnx/ebt312/bin/python q3_energy_landscape/analyze_q3.py \
  q3_energy_landscape/results/s2_step105000 q3_energy_landscape/results/s1_step105000
```
