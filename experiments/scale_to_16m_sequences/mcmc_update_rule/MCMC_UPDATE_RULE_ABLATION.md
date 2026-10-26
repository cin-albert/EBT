---
title: MCMC update-rule ablation for EBT S2
tags: [ebt, s2, mcmc, ablation, fixed-point]
---

# MCMC update-rule ablation for EBT S2

> [!summary] TL;DR
> - The public code updates the prediction as $z \leftarrow p - \alpha \nabla_p E$ with $p = \mathrm{softmax}(z)$. Its fixed points are stationary points of $\Phi = \alpha E - H - \tfrac12\lVert p\rVert^2$, not of $E$. The paper's pseudocode instead descends in logits.
> - We retrain xxs S2 with two alternative rules: **logits** ($z \leftarrow z - \alpha \nabla_z E(\mathrm{softmax}(z))$, the paper's pseudocode) and **log-probs** ($z \leftarrow \log p - \alpha \nabla_p E$, exponentiated gradient, i.e. mirror descent on $E$). Everything else matches the existing S2 105k/b32 baseline.
> - The logits rule risks near-zero first steps, so a short α pilot (3 × 2k steps, ~40 GPU-min) gates the full runs. Full runs: 105k steps, batch 32, GPU 5 (~10.5 h each). Evaluation: validation CE, Q2 (self-verification) and Q3 (energy landscape), against pre-registered hypotheses H1–H4.

## 1. Motivation

This is an ablation of the **MCMC update rule**. The transformer architecture does not change.

| Rule | Update (per token) | Fixed points | Source |
|---|---|---|---|
| `probs` (current) | $z_{t+1} = p_t - \alpha \nabla_p E(p_t)$ | stationary points of $\Phi = \alpha E - H(p) - \tfrac12\lVert p\rVert^2$ on the simplex | `model/nlp/ebt.py:132,153,171` |
| `logits` | $z_{t+1} = z_t - \alpha \nabla_z E(\mathrm{softmax}(z_t))$ | stationary points of $E \circ \mathrm{softmax}$ | paper, Appendix pseudocode (p. 44) |
| `log_probs` | $z_{t+1} = \log p_t - \alpha \nabla_p E(p_t)$ | stationary points of $E$ on the simplex | exponentiated gradient (mirror descent) |

Derivations: [[EXPLAIN_FIXED_POINT]] (`docs/EXPLAIN_FIXED_POINT.md`).

Points to keep in mind when reading results:
- **The paper's numbers come from the `probs` rule.** The authors' public code is what produced them. The `logits` rule matches the pseudocode, not the reported results, so there is no reason to expect it to land closer to the paper.
- **`logits` is not a pure minimizer of $E$ either.** $\nabla_z E = p \odot (g - \langle p, g\rangle)$ with $g = \nabla_p E$. It vanishes when $g$ is constant on the support of $p$, and it also shrinks as $p$ saturates toward one-hot, whatever $E$ is doing.
- **The three rules carry different amounts of memory.**
  - `probs`: feeding $p \in \Delta$ back as logits nearly resets the distribution, since $\mathrm{softmax}(p)$ changes any probability ratio by at most $e$. The next distribution is set almost entirely by $-\alpha g$.
  - `log_probs`: keeps the current distribution exactly and adds $-\alpha g$ to it, so the steps accumulate.
  - `logits`: also accumulates, but each step is scaled by $p$.

## 2. Risk: step size under the `logits` rule

- The initial condition is `random_noise`, so $p_0$ is close to uniform over $V = 50{,}277$ tokens and $p_i \approx 2 \times 10^{-5}$.
- Because $\nabla_z E$ is $\nabla_p E$ scaled by $p$, the first `logits` step at the same α is about $10^4$ times smaller than the `probs` step.
- With α = 5 fixed (S2 does not set `--mcmc_step_size_learnable`), the MCMC loop may barely move.
- The model can compensate by learning a much larger energy scale. That is the regime the paper reports as unstable with unnormalized inputs (§I.1.5: extreme activations, loss spikes).
- The `log_probs` rule keeps the gradient scale of `probs`, so it does not share this risk. Its accumulation (§1) may still call for a different α.

The pilot (§4.2) settles α before any long run.

## 3. Setup

| Item | Value |
|---|---|
| Workspace | `experiments/scale_to_16m_sequences/mcmc_update_rule/` (this file, the plan, scripts; `runs/` links to `/mnt/data/quangnx/research/ebt/scale_16m/mcmc_update_rule/runs`) |
| Model | xxs, EBT S2 only |
| Baseline | `ebt-s2-xxs-scaling-105k` (`experiments/reproduce_xxs_scaling`): valid loss 3.9431, ppl 51.58 at 105k, 1 GPU, 10h27m |
| Training length | 105k steps, batch 32, 1 GPU (GPU 5) |
| Data, split, seed | Same as the baseline: `reproduce_xxs_scaling/common.sh` (3.4M-doc `data/full`, its validation split, seed 33), so batch order matches |
| S2 hyperparameters | Same as the baseline (paper Table D.4): α = 5 randomized ×[½, 2], 2–3 steps, Langevin noise 3 added to $z$, replay buffer, `--truncate_mcmc`, `--no_mcmc_detach` |
| Changed | Only the update rule, plus α where the pilot requires it |

Notes on the setup:
- **Why 105k/b32 and not 250k/b64.**
  - 105k at batch 32 takes about 0.32 s/step × 105k ≈ 9–10 h. 250k at batch 64 on one GPU takes about 0.62 s/step × 250k ≈ 43 h.
  - The 105k baseline was trained on one GPU with the same data and order, so no baseline rerun is needed.
  - The 250k/b64 setting is worth running only if the 105k results show a clear difference.
- **Why the 3.4M build and not `full16m`.** `full16m` contains the 3.4M build as its first rows, but shuffling 16.1M documents gives a different batch order. Using the baseline's data directory keeps the comparison exact.

## 4. Plan

### 4.1 Code change

- Add a flag `--mcmc_update_rule {probs,logits,log_probs}`, default `probs`, so existing runs are unchanged.
- Apply it in the training/validation loop, `EBT_NLP.forward` in `model/nlp/ebt.py` (around `:115–186`), and in the analysis harness `experiments/thinking_study/common/harness.py` (`mcmc`), which Q2 and Q3 use.
- The advanced-inference loop (`_run_ebt_inference_steps`, `:442`) is not used by this study. It raises an error for any rule other than `probs` instead of being ported.
- How each rule is implemented:
  - `logits`: take the gradient with respect to the pre-softmax tensor (after noise), not the softmax output.
  - `log_probs`: the next logits are `log_softmax(z) - alpha * grad_p`.
- The replay buffer keeps storing the final logits $z$. This is valid for all three rules.

Tests (pytest):
- `probs` reproduces the current update bit-for-bit.
- `logits` equals `autograd.grad(E(softmax(z)).sum(), z)` applied with step α.
- At a point where $\nabla_p E$ is constant across the vocabulary, `log_probs` leaves $p$ unchanged.
- For every rule, `EBT_NLP.forward` and the harness loop give the same final prediction (S2 105k checkpoint, CPU, same seed).

Before the pilot, a 150-step `probs` gate run must reproduce the baseline's logged train loss, which confirms the refactor and the launch script.

### 4.2 Pilot (α gate)

The pilot only has to answer one question: does the MCMC loop do work under the new rule? That is visible early, so the pilot is kept small.

- 2k steps each, same setup and LR schedule as the full run, no validation, no checkpoints. At about 0.32 s/step plus start-up, each takes ~13 min; three take ~40 min on GPU 5.
- The measure is the **MCMC gain**: train CE after the first MCMC step minus train CE after the last step, averaged over the logged steps 1,500–2,000. Both are already logged (`train_initial_loss` is the CE after the first step; `train_final_step_loss`). Half of each train batch starts from the replay buffer, in the baseline as well, so the gains are comparable. The baseline's gain over the same steps is read from its `metrics.csv` (1.79 nats).

| Rule | α | Why |
|---|---|---|
| `logits` | 5 | same as the baseline |
| `logits` | 500 | about $10^2$× larger, to offset the ~$1/V$ scale of $p$ at initialization |
| `log_probs` | 5 | same as the baseline |

Pass criteria, per configuration:
1. No NaN/Inf in the train loss and the run completes.
2. MCMC gain ≥ 0.5 × the baseline's gain over the same steps.

Selection:
- `logits`: if both pass, the one with the lower mean `train_final_step_loss` over steps 1,500–2,000; if neither passes, report that as a result and do not run `logits` in full.
- `log_probs`: if α = 5 fails, run one more pilot at α = 1 before giving up on the rule.
- Learnable α is not piloted: it adds a run and a confound, and the fixed-α results already show whether step size is the problem.

### 4.3 Full runs

- One 105k run per rule, with the α chosen in the pilot, about 10.5 h each on GPU 5.
- Order: `logits` first; its Q2/Q3 evaluation and report section are finished before the `log_probs` run starts. The `log_probs` evaluation then extends the same report with the three-rule comparison.
- Checkpoints every 15k, all kept (the baseline kept only the best two, which were its last two).

## 5. Evaluation

### 5.1 Pre-registered hypotheses

| ID | Hypothesis | Measure | Baseline value (`probs`, 105k) |
|---|---|---|---|
| H1 | The new rule reaches validation CE ≤ baseline | final-step valid loss at 105k, same validation protocol (noise on, max steps) | 3.9431 |
| H2 | Thinking longer helps beyond the trained step count | Q3 CE after $k$ steps, $k = 1 \ldots 32$, noise 0 and 3: is the best $k$ greater than 3? | best at $k = 3$ |
| H3 | Energy is a better verifier | Q2 best-of-N at noise 3: CE of the lowest-energy pick minus CE of a random pick, N = 2…32 | −0.010 (N = 2), then worse than random: +0.006, +0.035, +0.068, +0.102 (N = 4…32) |
| H4 | The loop approaches a stationary point of its target | Q3 Part A: median over tokens of $\lVert d_{31}\rVert / \lVert d_{2}\rVert$, where $d$ is the rule's descent direction ($\nabla_p E$ projected onto the simplex for `probs`/`log_probs`, $\nabla_z E$ for `logits`) | 0.94 (noise 0), 0.91 (noise 3) |

- All runs have one training seed. Differences of a few hundredths of a nat in H1 are within the run-to-run noise we cannot measure here, and the report says so rather than claiming them.

### 5.2 Analyses

- **Training curves:** validation loss and per-step MCMC loss, with all three rules on one plot.
- **Q2 (self-verification):** `experiments/thinking_study/q2_verifier_quality`, same evaluation subsets as the 105k baseline.
- **Q3 (energy landscape):** `experiments/thinking_study/q3_energy_landscape`. `run_q3.py` currently hard-codes the `probs` map and $\Phi$ (lines 3–6, 105–111, 136), so it needs these changes:
  - make the map $T$ depend on `--mcmc_update_rule`;
  - line scans along the actual update direction of each rule;
  - Part A `gnorm` records the norm of the rule's descent direction (unchanged for `probs`);
  - the spectral radius of $T$ is computed only for `probs` (NaN otherwise); the Hessian spectrum of $E$ does not depend on the rule and is kept;
  - $\Phi$ is still recorded but interpreted only for `probs`.
- **OOD (optional):** the four OOD sets used for the 250k models, if H1 or H2 holds.

### 5.3 Deliverables

- `mcmc_update_rule/report/report.md`: results, H1–H4 verdicts and figures.
- Conclusion: does the update rule explain the differences between the paper's description and the code's behaviour (Φ-balance, best $k = 3$)?

## 6. Budget

| Stage | GPU 5 time |
|---|---|
| Code + tests (CPU) | — |
| Gate (150 steps) | ~3 min |
| Pilot (3 × 2k steps) | ~40 min |
| Full runs (2 × 105k) | ~21 h |
| Q2/Q3 evaluation (2 runs) | ~1 h (baseline Q2 took ~6 min per noise level, Q3 ~7 min per its last config.json) |
