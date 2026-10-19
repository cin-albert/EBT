---
title: "Thinking study — summarized findings (Q2, Q3; xxs, 105k steps)"
date: 2026-10-17
updated: 2026-10-18
tags: [ebt, thinking-study, summary, verifier, energy-landscape]
---

# Thinking study — summarized findings (Q2, Q3; xxs, 105k steps)

> [!summary] TL;DR
> - **Self-verification does not work at this scale.** The energy ranks candidates mostly by sharpness (lower E ⇒ lower entropy in 98% of pairs) and only weakly by correctness (62–63%). So best-of-N by lowest energy makes CE *worse* as N grows (S2, σ = 3: BoN-32 efficiency −0.20).
> - **"Thinking longer" does not help.** The public code's inference loop is a fixed-point iteration of Φ = αE − H − ½‖p‖², not GD on E. It converges and contracts, but CE is best at exactly the trained step count and gets worse with more steps (S2: 3.943 at k = 3 → 4.041 at k = 32).
> - **What survives scale is structural:** the update rule, the ≤ D-dimensional update subspace, and a loss that never trains the energy to rank candidates. The numbers are xxs-specific. Whether the verifier and the thinking penalty improve with data is being tested in [[IMPLEMENTATION_PLAN|scale_to_16m_sequences]].

## 1. Context

| Item | Value |
|---|---|
| Models | EBT-S1 (2 MCMC steps, learned α = 528.9), EBT-S2 (3 steps, α = 5, Langevin σ = 3); Transformer++ (T++) as reference |
| Size and training | xxs (D = 384), 105k steps × batch 32 = 3.36M sequences, one seed, one H100 NVL each |
| Validation CE at 105k (full validation split) | T++ 3.6964, S2 3.9431 (gap +0.247 nats) |
| Q2 eval | 1,024 validation sequences (228,522 tokens), 32 candidates per token |
| Q3 eval | Part A: 256 sequences (55,825 tokens), 32 steps. Part B: 32 sequences (6,949 tokens), exact 384 × 384 Hessians |
| Correctness gate | Passed (`common/results/gate.json`) |

Full reports: [Q2 report](q2_verifier_quality/report.md), [Q3 report](q3_energy_landscape/report.md) (figures in §3.5). Plan: [[THINKING_STUDY_PLAN]].

## 2. Findings, grouped by how far they generalize

### 2.1 Structural (follow from the code; independent of scale)

1. **The code's loop is a fixed point of Φ, not GD on E.**
   - The public code overwrites the logits with p = softmax(z), takes the gradient with respect to p, and updates z_{t+1} = p_t − α∇_pE(p_t) (`model/nlp/ebt.py:115-175`).
   - Its fixed points are the stationary points of Φ(p) = αE(p) − H(p) − ½‖p‖², an energy plus a hidden entropy term.
   - The paper's pseudocode (App. I.2, Listing 1) differs: it takes the gradient with respect to the logits z and accumulates z. This rule is in the code since the first public commit (`4aacd9e`), and no flag gives the paper's rule.
   - Empirical check: along the actual update, the line-minimizer of Φ is s = 1.0 at every measured state of both models, while E keeps decreasing to s = 2.
2. **The loop has short memory.**
   - Only p ∈ [0, 1] carries over between steps, so the initial noise is mostly forgotten.
   - Toy example: a token at p = 0.98 can be flipped to 0.377 by a single step, while mirror descent with the same gradient keeps 0.957.
   - Consequence for Q2: without Langevin noise, 32 candidates are nearly identical (CE std 0.001–0.002 nats), so there is nothing to verify.
3. **Every update lies in a ≤ D-dimensional subspace.**
   - The energy sees p only through e = Wp, with `vocab_to_embed` = `nn.Linear(V = 50,277, D = 384, bias=False)`. Hence ∇_pE = Wᵀ∇_eE, and updates live in the row space of W.
   - The relevant Hessian is therefore D × D per token, not V × V.
4. **The training loss never trains the energy to rank candidates.**
   - The loss is CE of the final prediction, backpropagated through the unrolled (truncated) steps. `contrastive_loss` is off in our runs.
   - Nothing pushes E(better candidate) below E(worse candidate) at equal sharpness, which is exactly where Q2 fails.
5. **`e_last_logged` is the wrong verifier quantity.**
   - The energy logged in training is taken before the final update. It ranks candidates *anti*-consistently (0.37–0.46 pairwise accuracy).
   - The energy of the returned prediction (`e_final`) must be used.

### 2.2 Likely robust (behavioural; needs confirmation at larger scale)

| Finding | Evidence (xxs, 105k) |
|---|---|
| Best CE at the trained step count; more steps hurt | S2 3.943 (k = 3) → 4.041 (k = 32), +0.098; S1 3.841 (k = 2) → 3.871, +0.030. Same with σ = 3 (3.955 → 3.992) |
| Energy is biased toward sharpness | Lower E ⇒ lower entropy in 98% of S2 σ = 3 pairs. One-hot of T++'s argmax gets the lowest energy in 44% (S2) / 53% (S1) of tokens, vs 30–36% for one-hot(target). Sharpening T++ (T = 0.5) lowers E while raising CE by 1.35 nats |
| Loop convergence and prediction quality are different targets | At the trained step count only 1.4–6.4% of tokens have converged, yet stopping there gives the lowest CE |
| Entropy is a better uncertainty signal than energy | AUROC for flagging top-1 errors: entropy 0.81–0.82 vs energy 0.62–0.68 |
| The loop's end points are not stationary points of E | α‖∇_pE‖ stays ≈1.2k from the trained state to t = 31 (ratio 0.94–1.01); no vanishing or exploding (p99/p1 ≈ 3). α∇E is balanced by the entropy term of Φ, not driven to 0 |
| Non-converged tokens are mostly in a period-2 cycle | 92% (S2) / 89% (S1) of tokens not converged at t = 31 flip-flop; S2 mean CE alternates 3.996 ↔ 4.041 at k = 31/32, so odd k is always better |
| Effective sharpness α·λ_max is the same for both models | ≈24 (S2) vs ≈27 (S1), although λ_max differs ≈100×; the learned energy scale compensates for α. α·λ_max ≫ 2, so plain GD on E would diverge; the softmax and entropy term keep the loop bounded |
| E is non-convex in the predicted embedding | 0% of tokens locally convex at every state; 33–56% negative Hessian eigenvalues at trained and late states. The loop still contracts because the entropy term in Φ is strongly convex. S2 moves from concave noise (trace < 0 for 99.8% of tokens) to a valley (a few stiff directions, a large flat bulk); S1 stays a balanced saddle (\|λ_min\| ≈ 0.9 λ_max) |

### 2.3 Measured values (scale-dependent; read as xxs numbers only)

**Q2 — verifier quality**

| Metric | S1 105k, σ = 0 | S2 105k, σ = 0 | S2 105k, σ = 3 | S2 90k, σ = 3 |
|---|---|---|---|---|
| Single-sample CE | 3.8152 | 3.9103 | 3.9224 | 3.9572 |
| BoN-32 CE (lowest `e_final`) | 3.8151 | 3.9099 | **4.0232** | 4.1205 |
| Oracle-32 CE | 3.8104 | 3.9077 | 3.4138 | 3.4225 |
| Verifier efficiency @32 | 0.02 | 0.16 | **−0.20** | −0.31 |
| Pairwise: lower E ⇒ lower CE (\|ΔCE\| > 0.1) | 0.389 | 0.304 | **0.623** | 0.592 |
| Pairwise: lower E ⇒ lower entropy | 0.727 | 0.736 | **0.982** | 0.981 |

- With σ = 3, energy selection helps only at N = 2 (−0.010 nats) and hurts from N = 4 on (+0.006, +0.035, +0.068, +0.102 at N = 4–32).
- Picking the *highest* energy is also worse than random (4.0933 at N = 32), so CE is U-shaped in energy.
- 90k → 105k improves the verifier slightly (efficiency −0.31 → −0.20), in the direction of the paper's Fig. B.1. These are two checkpoints of one run, not a trend.

**Q3 — landscape and update map**

| Metric | S2 (trained state) | S1 (trained state) |
|---|---|---|
| Median spectral radius ρ of the update Jacobian (t = 0 → trained → t = 31) | 1.39 → 0.67 → 0.49 | 0.42 → 0.35 → 0.35 |
| Tokens converged within 32 steps (noise 0) | 82.6% | 96.9% |
| Spearman(steps to converge, entropy) | 0.81 | 0.18 |
| Participation ratio of ∇_eE (of 384) | 20.1 | 28.7 |
| λ_max / median\|λ\| of ∇²_eE | 75 | 21 |
| cos(g_t, g_{t−1}) after step 1 | 0.998–1.0 | 0.998–1.0 |
| Spearman(ΔE, ΔCE) per step | ≈ 0.4 | 0.55 → 0.14 → 0.07 → 0.0 |
| Line argmin s along the update: CE / E / Φ | 1.6 / 2.0 / 1.0 | 1.25 / 2.0 / 1.0 |

- S2 oscillates for the first steps (CE 3.943, 4.078, 3.972, 4.060 at k = 3–6), consistent with ρ > 1 at the start.
- S2 with σ = 3 never reaches a fixed point (TV ≈ 0.89 per step; 4.6% of tokens settle).
- Adaptive compute (the paper's Facet 1) holds for S2 without noise: hard, high-entropy tokens take longer to converge.

## 3. Answers to the study questions

| Question | Answer at xxs / 105k |
|---|---|
| **Q2:** Is lower energy a better prediction? | Only weakly. Energy ranks by sharpness, not correctness. Best-of-N by energy is net-harmful, and entropy is the better uncertainty signal. The "+ self-verification" arm of the paper's thinking claim fails in-distribution here |
| **Q3:** What does inference optimize, and does more of it help? | In the public code, a fixed point of Φ (E plus a hidden entropy term), not E. It converges and contracts, but loss is best at the trained step count. The landscape is non-convex, anisotropic and effectively ~20–30-dimensional. Its end points are not stationary points of E, and most non-converged tokens sit in a period-2 cycle |

## 4. Implications (hypotheses, not results)

| Lever | Supported by | Write-up |
|---|---|---|
| Train the energy to rank candidates (contrastive / ranking loss with model-generated negatives) | Structural finding 4; Q2 sharpness bias | [[IDEA_LOW_DIM_SPACE_SEARCH]] §4 |
| Update-rule ablation: code (p-gradient) vs paper (z-gradient) vs mirror descent | Structural findings 1–2 | [[THINKING_STUDY_PLAN]] M4 |
| Second-order step in the ≤ D-dim subspace (saddle-free truncated Newton) | Findings 3; anisotropy 21–75; participation ratio 20–30 | [[IDEA_SECOND_ORDER_DERIVATIVE]] |
| Gradient-free search in a ~20–30-dim subspace | Same, plus a fixed basis shared across tokens | [[IDEA_LOW_DIM_SPACE_SEARCH]] |
| Per-step or per-token α / stopping rule | Φ line-optimal at s = 1, CE prefers 1.25–1.75; steps-to-converge tracks entropy (0.81, S2) | [Q3 report](q3_energy_landscape/report.md) §4 |
| Damping, p ← ½(p + T(p)) | Period-2 cycle in 16% of S2 tokens; α·λ_max ≈ 24 | [Q3 report](q3_energy_landscape/report.md) §3.4 |
| Momentum / Nesterov | Gradients already collinear, map already contracts | Low expected gain |

Search and second-order methods that minimize the *current* energy harder should make predictions worse (Q2). They only make sense together with a ranking-trained energy.

## 5. Open: does scale change the picture?

The 105k runs are about 38× below the paper's data budget (3.36M vs 128M sequences). The [[IMPLEMENTATION_PLAN|scale_to_16m_sequences]] experiment (T++ and S2, batch 64, 250k steps = 16M sequences) pre-registers three checks against this summary:

| Claim | Reference value here | Pass condition at 16M |
|---|---|---|
| S2 − T++ gap shrinks | +0.247 nats | Gap below 0.247 by more than 0.03 nats |
| Verifier improves | BoN-32 efficiency −0.20; pairwise consistency 0.623 | Both improve |
| Thinking penalty shrinks | +0.098 nats (k = 3 → 32) | Smaller penalty |

Limitation for that comparison: the 105k point used batch 32 and the 16M runs use batch 64. At equal sequences seen (~3.2M), T++ at batch 64 has validation loss 3.825 vs 3.696 at batch 32. So within-run curves (25k–250k at batch 64) are the cleaner comparison.

> [!warning] Scope
> - xxs only, one seed per model, in-distribution validation only. OOD behaviour is untested.
> - Mechanism claims (Φ, entropy term, short memory) apply to the public code's update rule, not to the paper's pseudocode.
> - Landscapes were co-trained with this update rule. A different inner optimizer would learn a different landscape, so §4 states plausibility only.
