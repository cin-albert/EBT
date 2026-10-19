---
title: "Energy landscape — visualization plan (Q3)"
date: 2026-10-18
tags: [ebt, thinking-study, energy-landscape, visualization]
---

# Energy landscape — visualization plan (Q3)

> [!summary] TL;DR
> - Goal: make the Q3 findings readable as figures. Every chart plots the **inference loop** (32 steps, S1 and S2 at 105k), not the training trajectory, and uses the update rule of the public code, z ← p − α∇_pE(p) (see [[q3_energy_landscape/report|Q3 report]] §2).
> - 8 charts. 6 are drawable from the saved arrays (`results/*/part_a_noise*.npz`, `part_b.npz`). 2 need a small Part B re-run that saves Hessian eigenvectors and per-step gradients in e-space.
> - Dropped from the first draft: the edge-of-stability threshold 2/η (the loop is not GD on E, so the relevant criterion is ρ(J_T) < 1), the EMA of cos(g_t, g_{t−1}) (32 steps, already ≈1), and a correlation heatmap of two scalars (replaced by a density plot).

## 1. Conventions

| Item | Choice |
|---|---|
| x-axis "step" | Inference step t = 0…31 (t = 0 is the initial noise). Mark the trained step count with a vertical line (S1: 2, S2: 3) |
| Models | S1 and S2 in separate panels or colors. S2 noise 0 is the main curve; S2 noise 3 is optional |
| Aggregation over tokens | Median line + p10–p90 band. Not the mean: the distributions are skewed |
| Scale | S1 and S2 differ by ≈100× in ‖g_p‖ and λ. Use log-y, or plot the α-scaled quantity (α‖g_p‖, αλ), which is what the update actually sees |
| Gradient | g_p = ∇_pE with the softmax-null component removed (what Part A stores). Hessian quantities are in e = W_v p (384-d) |

## 2. Charts

### 2.1 Slope: ‖g‖ by inference step — keep, revised

- **Plot:** median and p10–p90 band of α‖g_p‖ vs t, one line per model.
- **Change from draft:** the draft grouped steps into "2 or 3 steps per loop". That mixes up training (S1: 2 steps; S2: random 2–3) with inference. Here every token runs the same 32 inference steps, so there is nothing to group; the trained step count is a vertical marker.
- **What it should show:** no vanishing or exploding (spread ≈3×), and **no decay toward 0 at convergence**, i.e. the loop's end points are not stationary points of E (report §3.4).
- **Data:** Part A `gnorm`. Available.

### 2.2 Gradient direction — keep the line, drop the EMA, replace the Gram matrix

- **Plot A:** median and p10–p90 of cos(g_t, g_{t−1}) vs t.
- **Interpretation, corrected:** this loop overwrites the state instead of accumulating it, so cos ≈ 1 means *the landscape presents the same direction again*, which is what happens near a fixed point. It is not evidence of a "straight trajectory". A negative value means the gradient reverses (zig-zag), not "horizontal oscillation".
- **Expected:** ≈0.97 at t = 1 and 0.998–1.0 afterwards. The chart is nearly flat, so keep it small. An EMA over 32 points adds nothing.
- **Plot B (replaces the Gram matrix):** median of cos(Δp_t, Δp_{t−1}) with Δp_t = p_{t+1} − p_t. If g barely changes direction but p flips, this goes to −1 and shows the **period-2 cycle** directly (16% of S2 tokens, report §3.4). A T×T Gram matrix of g would be almost all ones.
- **Data:** Plot A from Part A `cos_prev`, available. Plot B needs Δp vectors, which are not saved. Add to the re-run (§3).

### 2.3 Curvature and stability — revised

The draft's line "λ_max vs step with a horizontal line at 2/η" tests edge of stability (EoS). EoS describes **GD on a training loss over training iterations** with a fixed learning rate η. It does not transfer here:

- the loop is a softmax fixed-point map, not GD on E. Measured α·λ_max ≈ 24 ≫ 2, yet the loop stays bounded;
- the stability criterion of the map is the spectral radius ρ of its Jacobian J_T (< 1 means contracting), which Part B already measures.

Replacement charts:

- **Plot A — ρ(J_T) distribution** (histogram or violin) at t = 0, trained state, t = 31, with a vertical line at ρ = 1. Shows the move from expanding (S2 t = 0: median 1.39) to contracting (0.49). Data: Part B `rho`. Available.
- **Plot B — eigenvalue spectrum of ∇²_eE**: density of λ / λ_max per token, pooled, at the three states. Shows saddle shape (negative mass), the flat bulk and the few stiff directions better than a single λ_max line. Data: Part B `eig`. Available.
- **Plot C — gradient–curvature alignment** (keeps the draft's second idea): share of ‖∇_eE‖² in the top-k positive eigenvectors and in the negative eigenspace, vs t. Use the squared projection, because eigenvector sign is arbitrary and a single cos with the top eigenvector ignores near-ties. Answers whether the gradient pushes along stiff directions (preconditioning helps) or negative ones (E keeps falling, Q2 sharpness bias). Data: needs eigenvectors and g_e. Re-run (§3).
- **Optional:** λ_max at every step (not just three states) is possible on fewer tokens; each state costs 384 Hessian–vector products per token.

### 2.4 Local shape: 1D slice along the update — keep, specified

- **Plot:** for each of the three states, ΔE(s) = E(s) − E(0), ΔΦ(s) and ΔCE(s) along z = p − s·α∇_pE, s ∈ [0, 2]. Median + IQR over tokens, vertical line at s = 1 (the actual update).
- **What it should show:** Φ is minimized at s = 1, CE at s ≈ 1.25–1.75, and E keeps decreasing to s = 2.
- **Normalize per token** (subtract the value at s = 0, optionally divide by |E(2) − E(0)|). Otherwise S1 and S2 cannot share an axis.
- **Optional 2D slice:** update direction × top Hessian eigenvector, for a handful of example tokens. Needs eigenvectors (§3).
- **Data:** Part B `line_energy`, `line_phi`, `line_logp`. Available.

### 2.5 Energy vs entropy — replace the heatmap

A correlation heatmap of two scalars is a single number. Two different questions are involved:

- **Across tokens** (is energy a difficulty signal?): hexbin density of final energy vs entropy, one panel per model, with Spearman ρ in the title. Color by mean CE per bin. Data: Part A `energy`, `entropy`, `logp_final`. Available.
- **Within a token, across candidates** (Q2's sharpness bias): scatter of ΔE vs Δentropy over candidate pairs, colored by sign of ΔCE. 98% of S2 σ = 3 pairs fall in the "lower E ⇒ lower entropy" quadrants. Data: Q2 `results/*/arrays.npz`. Available.

### 2.6 Added: thinking-longer trajectories

Not in the draft, but it is the chart that links landscape to loss.

- **Plot:** mean CE, E, Φ and entropy vs k = 1…32 (four small panels), trained step count marked.
- **What it should show:** CE is best at the trained step count; S2 alternates between odd and even k (3.996 ↔ 4.041 at k = 31/32), i.e. the 2-cycle at population level; Φ is flat while E oscillates.
- **Data:** Part A `metrics.json` `mean_per_step` and `ce_after_k_steps`. Available.

## 3. Summary and re-run needed

| # | Chart | Status vs draft | Data |
|---|---|---|---|
| 2.1 | α‖g_p‖ vs t | Revised (no step grouping) | Available |
| 2.2A | cos(g_t, g_{t−1}) vs t | Kept, interpretation corrected, EMA dropped | Available |
| 2.2B | cos(Δp_t, Δp_{t−1}) vs t | Replaces Gram matrix | Re-run |
| 2.3A | ρ(J_T) distribution | Replaces 2/η line | Available |
| 2.3B | Hessian spectrum density | New | Available |
| 2.3C | Gradient energy in stiff / negative eigenspaces | Revised from cos(g, top eigvec) | Re-run |
| 2.4 | 1D slices of E, Φ, CE | Kept, specified | Available |
| 2.5 | Energy–entropy density (across tokens, within token) | Replaces heatmap | Available |
| 2.6 | CE/E/Φ/entropy vs k | New | Available |

Re-run: Part B only (32 sequences, about the same cost as the original Part B). Use `torch.linalg.eigh` instead of `eigvalsh` to keep eigenvectors, and save g_e and Δp at every step. Store only the top and bottom ~20 eigenvectors per token to keep the files small.

> [!warning] Scope
> - All charts describe xxs / 105k / one seed. They illustrate the numbers in the Q3 report; they do not add new evidence beyond §2.2B and §2.3C.
> - A training-time EoS study (sharpness of the training loss vs training step) is a different question and would need many checkpoints; we only have 90k and 105k.
