---
title: MCMC update-rule ablation for EBT S2 — report
tags: [ebt, s2, mcmc, ablation, report]
---

# MCMC update-rule ablation for EBT S2 — report

> [!summary] TL;DR
> - **`logits` (the paper's pseudocode rule) does not train** at xxs scale: across α = 5, 500 and 50,000, with and without Langevin noise, the MCMC loop never improves the prediction (gain ≤ 0.04 nats at 2k steps, vs 1.79 for the `probs` baseline). It was not run to 105k.
> - **The baseline's Langevin noise (std 3 on logits) is incompatible with both alternative rules**: they keep the logits between steps, so the noise accumulates and CE rises above that of a uniform guess. With noise 0, `log_probs` passes the pilot (gain 1.14).
> - `log_probs` (exponentiated gradient, α = 5, noise 0) is training to 105k; its H1–H4 evaluation is pending (Tasks 9–10).

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

Pending: the 105k run (α = 5, noise 0) is training; Q2/Q3 and the H1–H4 table follow in Tasks 9–10.

## 4. Reproduction

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
```
