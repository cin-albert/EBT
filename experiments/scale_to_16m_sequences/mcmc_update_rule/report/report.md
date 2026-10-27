---
title: MCMC update-rule ablation for EBT S2 — report
tags: [ebt, s2, mcmc, ablation, report]
---

# MCMC update-rule ablation for EBT S2 — report

> [!summary] TL;DR
> - **Neither alternative rule beats the repo's `probs` update at xxs scale.** `logits` (the paper's pseudocode) does not train: the MCMC loop never moves the prediction (gain ≤ 0.04 nats vs 1.79). `log_probs` (exponentiated gradient, noise 0) trains, but worse and **unstably**: valid loss 5.46 at its best (30k, baseline 4.40) and 18.0 at 105k (baseline 3.94), with recurring loss spikes from ~35k.
> - Its energy loop behaves more like the paper describes in two ways: thinking past the trained 2–3 steps helps a little (best k = 5 at 30k), and the lowest-energy candidate beats a random one at every N. Its smaller ‖d₃₁‖/‖d₂‖ (0.51 vs 0.94) holds by construction and a rule-neutral convergence measure shows no advantage. All of this is measured on a much worse model without Langevin noise, so it does not show that the rule would help a model of baseline quality.
> - The baseline's Langevin noise (std 3) cannot be reused with rules that keep the logits: the noise accumulates and CE rises above a uniform guess. Both alternative rules were therefore run with noise 0, a deviation from the baseline.

Spec: [[MCMC_UPDATE_RULE_ABLATION]]. Plan: [[IMPLEMENTATION_PLAN]].

## 1. Setup

- Model, data, split, seed, batch 32, LR schedule and all other S2 flags are those of the baseline run `ebt-s2-xxs-scaling-105k` (`experiments/reproduce_xxs_scaling`; valid loss 3.9431 at 105k).
- The rule is chosen with `--mcmc_update_rule {probs,logits,log_probs}` (`model/nlp/mcmc_update.py`); the default `probs` is the repo's update.
- **Refactor gate.** A 150-step `probs` run with the new code reproduces the baseline's train loss bit for bit (31.4312 / 13.4261 / 10.1670 at steps 49 / 99 / 149; `results/gate.json`).
- **Pilot measure.** MCMC gain = train CE after the first MCMC step minus train CE after the last step, averaged over logged steps 1,500–2,000. Pass: no NaN, run complete, gain ≥ 0.5 × the baseline's gain over the same steps (0.5 × 1.79 = 0.90).

## 2. Pilot results

Means over steps 1,500–2,000 of 2k-step pilots (`results/pilot.json`, `pilot-n0.json`, `pilot-logits-a50000-n0.json`). "Energy gap" is the logged mean energy evaluated at the first MCMC step minus that at the last step (`train_initial_final_pred_energies_gap`); positive means the loop lowers the energy.

| Rule | α | Noise | CE after step 1 | CE after last step | Gain | Energy gap | Pass |
|---|---|---|---|---|---|---|---|
| `probs` (baseline) | 5 | 3 | 8.75 | 6.96 | **+1.79** | 1.174 | — |
| `logits` | 5 | 3 | 18.70 | 23.54 | −4.84 | −0.850 | no |
| `logits` | 500 | 3 | 18.71 | 23.55 | −4.84 | −0.007 | no |
| `log_probs` | 5 | 3 | 13.47 | 15.34 | −1.88 | 0.188 | no |
| `logits` | 5 | 0 | 11.31 | 11.30 | +0.01 | 0.217 | no |
| `logits` | 500 | 0 | 11.32 | 11.30 | +0.02 | 0.010 | no |
| `logits` | 50,000 | 0 | 11.27 | 11.23 | +0.04 | 0.000 | no |
| `log_probs` | 5 | 0 | 8.85 | 7.71 | **+1.14** | 0.320 | **yes** |

For reference, a uniform prediction over $V = 50{,}277$ tokens has CE $\log V = 10.83$.

### 2.1 Langevin noise accumulates under rules that keep the logits

- Each MCMC step adds $\mathcal N(0, 3^2)$ to the logits $z$ before the softmax.
- Under `probs`, the next logits are $p - \alpha g$ with $p \in [0, 1]$: the noise of the previous step is discarded.
- Under `logits` ($z - \alpha \nabla_z E$) and `log_probs` ($\log p - \alpha g$, and $\log \mathrm{softmax}(z)$ keeps $z$ up to a constant), the noise is carried into the next step. The replay buffer (half of each train batch) carries it across iterations as well.
- For logits with i.i.d. $\mathcal N(0, \sigma^2)$ entries, the expected CE is about $\log V + \sigma^2/2$. Starting from $\sigma^2 = 1$ (the random initial condition) and adding variance 9 per step gives about 15.8 after one step and 20.3 / 24.8 after two / three steps if the gradient does nothing.
- This matches the noise-3 `logits` runs (18.7 after step 1, with replay-buffer starts that are already noisier; 23.5 after the last step), and the noise-0 runs sit at $\log V + 1/2 = 11.33$, the CE of the initial condition.
- Consequence: the noise level tuned for `probs` cannot be reused for the other two rules. The comparison below uses noise 0 for them, which is a **deviation from the baseline** (noise 3): S2's stochastic sampling is switched off for the new rule.

### 2.2 `logits` does not move the prediction

- With noise 0, CE stays at the initial condition's 11.3 for all three α values. Gains of 0.01–0.04 nats are about 2% of the baseline's.
- Mechanism (spec §2): $\nabla_z E = p \odot (\nabla_p E - \langle p, \nabla_p E\rangle)$, and at the near-uniform start $p_i \approx 1/V \approx 2 \times 10^{-5}$, so a step is about $10^4$× smaller than a `probs` step at the same α.
- Raising α does not fix this. At α = 50,000 the first steps overshoot (train CE 725 → 1,498 at step 49), after which the model learns an energy that is flat along the update: the logged energy gap falls to 0.000 and CE returns to 11.3.
- Verdict: **`logits` was dropped before the full run** (spec §4.2 selection). H1–H4 are not evaluated for it.
- Not tested: a learnable α, an α schedule, or a different initial condition (for example a peaked one, where $p$ is not tiny). The negative result holds for fixed α with the baseline's random-noise start.

## 3. `log_probs` vs `probs`

Run: `s2-rule-log-probs-105k`, α = 5, noise 0, otherwise the baseline's flags. A first attempt hung at the 15k validation (a DataLoader worker deadlocked; no checkpoint was saved) and was restarted from scratch; the restart reproduces its train loss bit for bit.

### 3.1 Training is unstable

| Step | 15k | 30k | 45k | 60k | 75k | 90k | 105k |
|---|---|---|---|---|---|---|---|
| `log_probs` valid loss | 5.59 | **5.46** | 8.28 | 6.87 | 9.90 | 7.54 | 18.01 |
| `probs` baseline valid loss | 4.71 | 4.40 | 4.25 | 4.14 | 4.04 | 3.98 | 3.94 |

- Up to ~30k the run learns steadily but more slowly than the baseline (train CE after the last step 6.0 vs 4.8 at 15k–20k).
- From ~35k, train loss spikes recur and grow: the mean train CE after the last step rises to 11.0 over 60k–65k, with a single-batch peak of 1,022 at ~66k. The run recovers to ~6.2 over 67k–70k, then spikes persist (mean train CE after the last step 9.4–24.5 per 5k bin over 70k–105k); the 105k checkpoint sits inside a spike (valid CE after the first step 107, after the last 18.0).
- Gradient clipping almost never fires (logged clipping fraction 0.01% on average, 13% at most, at the 66k spike), and the logged energy gap grows with the spikes (mean 0.4 over 0–5k, 1.7 over 60k–65k, 3.2 over 100k–105k).
- Likely mechanism (not verified): `log_probs` adds each step to the current log-probabilities with no reset, so once the energy gradient grows, α∇E can push the prediction to a confident wrong distribution. Under `probs` the next logits are $p - \alpha g$ with $p \in [0,1]$, which does not carry the previous step's scale. Confirming this needs ‖α∇E‖ and the entropy of $p$ around 60k–66k.
- The pilot's "MCMC gain" is misleading here: over logged steps 69,000–69,999 it was 2.485 nats (baseline 2.482 over 104,000–104,999), but only because the CE after the first step had exploded.

### 3.2 Hypotheses

The 105k checkpoint is the pre-registered one. Because it is mid-spike, the 30k checkpoint (best validation loss) is also evaluated, as a supplementary row. Q2 is run at noise 0 and H3 is read at each run's training noise (baseline 3, `log_probs` 0); Q3 Part A for `log_probs` covers noise 0 only.

| Rule | H1 valid loss | H2 best k (CE), noise 0 | H2 best k (CE), noise 3 | H3 energy − random, N = 2 / 4 / 8 / 16 / 32 | H4 median ‖d₃₁‖/‖d₂‖, noise 0 / 3 |
|---|---|---|---|---|---|
| `probs` baseline (noise 3) @105k | 3.9431 | 3 (3.9434) | 3 (3.9545) | −0.010 / +0.006 / +0.035 / +0.068 / +0.102 | 0.94 / 0.91 |
| `log_probs` (noise 0) @105k | 18.0054 | 8 (15.67) | — | −0.143 / −0.224 / −0.336 / −0.468 / −0.643 | 0.59 / — |
| `log_probs` (noise 0) @30k, supplementary | 5.4582 | 5 (5.3167) | — | −0.022 / −0.036 / −0.045 / −0.053 / −0.057 | 0.51 / — |

Source: `report/compare.md`; figure `report/figures/ce_vs_k.png`.

![[figures/ce_vs_k.png]]

Q3 CE after k steps, noise 0:

| k | 1 | 2 | 3 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| `probs` baseline @105k | 6.28 | 4.14 | **3.94** | 4.08 | 4.05 | 4.04 | 4.04 |
| `log_probs` @30k | 6.69 | 5.68 | 5.42 | 5.34 | 5.40 | 5.87 | 7.00 |
| `log_probs` @105k | 105.4 | 22.9 | 18.0 | 16.7 | **15.7** | 16.8 | 21.9 |

(`log_probs` @30k reaches its minimum 5.317 at k = 5.)

Verdicts:
- **H1 (valid CE ≤ baseline): rejected.** 18.0 vs 3.94 at 105k; even the best checkpoint (5.46 at 30k) is 1.06 nats above the baseline at the same step.
- **H2 (best k > 3): holds, weakly.** Best k = 5 at 30k (CE 0.10 below k = 3) and 8 at 105k; the baseline's best is 3. Past k ≈ 8, CE rises again (to 7.0 at k = 32 for 30k), so longer thinking helps only a little beyond the trained count, then hurts.
- **H3 (energy is a better verifier): holds in sign.** The lowest-energy candidate beats a random one at every N, and the margin grows with N; the baseline's energy pick is worse than random from N = 4. The effect is small at 30k: at N = 32 it recovers 0.057 of the 1.21-nat gap between random and oracle (verifier efficiency 0.047).
  - Candidate diversity differs by construction. At noise 0, `log_probs` candidates differ only through the random initial condition, which the rule keeps (mean JS between candidates 0.10). `probs` discards it, so the baseline needs noise 3 for diversity (0.028; 8.5·10⁻⁷ at noise 0).
- **H4 (loop approaches a stationary point): holds as measured, but the measure favours `log_probs` by construction.** The median ‖d₃₁‖/‖d₂‖ is 0.51 (30k) and 0.59 (105k) vs 0.94.
  - $d$ is the centred $\nabla_p E$, which vanishes at E-stationary points: the fixed points of `log_probs`, but not of `probs`, whose fixed points are stationary for $\Phi$ (spec §1). A `probs` loop sitting exactly at its own fixed point would still score near 1, so 0.51 vs 0.94 compares the rules' targets as much as their convergence.
  - A rule-neutral measure, the mean total-variation change between consecutive predictions (`tv_next`, Q3 Part A, noise 0), shows no better convergence for `log_probs`: at k = 31 it is 0.035 for the baseline, 0.037 at 30k (lowest 0.031 at k ≈ 16, then rising) and 0.108 at 105k. The baseline's change keeps falling up to k = 31.
  - CE also rises after k ≈ 5–8 while $d$ keeps shrinking, so the E-stationary points the loop approaches are not low-CE points.

Caveats:
- One seed per run.
- `log_probs` runs without Langevin noise; a `probs` run with noise 0 was not trained, so the rule and the noise level are confounded in every comparison.
- H2–H4 are measured on a much worse model (5.46 vs ~4.0); verifier and landscape properties may change with quality.
- Not done from spec §5.2: the training-curve plot (replaced by the tables in §3.1) and the optional OOD evaluation (its trigger, H1 or H2 holding, is met only by the weak H2 result on a model 1.5 nats worse than the baseline, so it was skipped).

## 4. Conclusion

- The spec asked whether the update rule explains the gap between the paper's description and the code's behaviour (Φ-balance, best k = 3, energy failing as a verifier).
- **`logits`** (the pseudocode rule) cannot be the explanation in the direction of "the paper's rule works better": with a fixed α it does not train at all. Its step is $p_i$ times the `probs` step, so no fixed α suits both the near-uniform start ($p_i \approx 1/V$) and a concentrated prediction; `log_probs` is exactly `logits` with that factor divided out.
- **`log_probs`** (the rule whose fixed points are stationary points of $E$) shows two behaviours the paper describes and the baseline lacks: longer thinking helps a little, and energy ranks candidates better than chance. Its loop moves closer to an E-stationary point, as the rule is designed to, but by a rule-neutral measure it does not settle better than the baseline (§3.2, H4). This is consistent with the `probs` rule's Φ-balance being part of why the baseline stops at k = 3 and why its energy fails as a verifier, but the noise confound (§3.2 caveats) prevents a firmer claim.
- It does not reproduce the paper's quality: it trains to a worse loss and becomes unstable. At this scale and α, the repo's `probs` rule is the better choice for language modelling CE.
- Untested next steps, in order of cost: a `probs` run with noise 0 (separates rule from noise); `log_probs` with a smaller α or a clamp on α∇E (tests the instability mechanism); learnable α for `logits`.

## 5. Reproduction

From `experiments/scale_to_16m_sequences/mcmc_update_rule/`:

```bash
# refactor gate + first pilot round (noise 3)
GPU_ID=5 ./pilot.sh
# second pilot round (noise 0)
GPU_ID=5 SKIP_GATE=1 TAG=-n0 ./pilot.sh --langevin_dynamics_noise 0
# logits at alpha 5e4, noise 0
GPU_ID=5 ./run_rule.sh logits 50000 pilot-logits-a50000-n0 --max_steps 2000 --limit_val_batches 0 \
  --save_top_k_ckpts 0 --langevin_dynamics_noise 0
/mnt/data/conda-envs/quangnx/ebt312/bin/python check_runs.py pilot runs/logs/csv/pilot-logits-a50000-n0/version_0/metrics.csv
# full log_probs run
GPU_ID=5 ./run_rule.sh log_probs 5 s2-rule-log-probs-105k --save_top_k_ckpts -1 --langevin_dynamics_noise 0
# Q2/Q3 for the pre-registered 105k checkpoint and the supplementary 30k one
GPU_ID=5 ./eval.sh s2_rule_log_probs
GPU_ID=5 STEP=30000 ./eval.sh s2_rule_log_probs
/mnt/data/conda-envs/quangnx/ebt312/bin/python compare.py   # -> report/compare.md, report/figures/ce_vs_k.png
```
