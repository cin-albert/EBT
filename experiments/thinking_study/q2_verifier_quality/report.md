---
title: "Q2 — Is lower energy a better prediction? (verifier audit)"
date: 2026-10-15
tags: [ebt, thinking-study, verifier, self-verification]
---

# Q2 — Is lower energy a better prediction? (verifier audit)

> [!summary] TL;DR
> - **Self-verification hurts at this scale.** For S2 with its training noise (σ = 3), picking the lowest-energy of N candidates makes CE *worse* as N grows: −0.010 nats at N = 2, then +0.006, +0.035, +0.068 and +0.102 nats at N = 4, 8, 16, 32 (verifier efficiency −0.20 at N = 32). Without noise, S1 and S2 produce near-identical candidates (CE std ≈ 0.001–0.002 nats), so there is nothing to choose from.
> - **The energy mostly measures sharpness, not correctness.** Within a token, the lower-energy candidate is also the lower-entropy one in 98% of pairs (S2, σ = 3). It ranks candidates by CE correctly in only 62% of pairs (chance = 50%; VFScale reports ≈0.73 for its naive energy). The sharpened one-hot of Transformer++'s argmax gets the lowest energy more often (44–53% of tokens) than the one-hot of the true token (30–36%).
> - **As an uncertainty signal, energy is weaker than plain predictive entropy:** AUROC for flagging top-1 errors 0.62–0.68 vs 0.81–0.82. Verification improves slightly with training (S2 BoN-32 efficiency −0.31 at 90k → −0.20 at 105k), which matches the direction of the paper's Fig. B.1, but it stays net-harmful.

## 1. Setup

- Checkpoints: S1 and S2 at 105k and 90k; Transformer++ at 105k for off-trajectory candidates (see [[THINKING_STUDY_PLAN]] §2). The correctness gate passed (`../common/results/gate.json`).
- Eval set: 1,024 fixed validation sequences (228,522 target tokens; seed 0 subset of the 33,320-doc split). On this subset, Transformer++ CE = 3.6706.
- **Candidates:** G = 32 per token from the model's own inference at its validation step count (S1: 2 steps, S2: 3), each with a fresh initial-noise draw, plus Langevin noise at σ = 3 (S2's training value) or σ = 0.
- **Candidate energy:** `e_final` = E(softmax(z_N)) on the last landscape, i.e. the energy of the prediction actually returned. We also report `e_last_logged`, the energy at the last pre-update state, which is what training logs.
- **Off-trajectory candidates**, scored by the same energy: Transformer++ softmax at T = 0.5/1/2, one-hot(target), one-hot(Transformer++ argmax), and uniform.
- Code: `run_q2.py`, `analyze_q2.py`. Raw arrays: `results/<run>/arrays.npz` (not in git); metrics: `results/<run>/metrics.json`.

## 2. Results

### 2.1 Candidate diversity and best-of-N

| Run | Single-sample CE | CE std across 32 candidates | BoN-2 | BoN-4 | BoN-8 | BoN-16 | BoN-32 | Oracle-32 | Efficiency @32 |
|---|---|---|---|---|---|---|---|---|---|
| S1 105k, σ=0 | 3.8152 | 0.0023 | 3.8152 | 3.8152 | 3.8151 | 3.8151 | 3.8151 | 3.8104 | 0.02 |
| S1 90k, σ=0 | 3.8425 | 0.0023 | 3.8424 | 3.8424 | 3.8424 | 3.8424 | 3.8424 | 3.8375 | 0.01 |
| S2 105k, σ=0 | 3.9103 | 0.0012 | 3.9102 | 3.9101 | 3.9100 | 3.9100 | 3.9099 | 3.9077 | 0.16 |
| **S2 105k, σ=3** | 3.9224 | 0.3048 | **3.9125** | 3.9275 | 3.9562 | 3.9888 | **4.0232** | 3.4138 | **−0.20** |
| S2 90k, σ=3 | 3.9572 | 0.3194 | 3.9647 | 3.9938 | 4.0352 | 4.0779 | 4.1205 | 3.4225 | −0.31 |

BoN-N = mean CE when picking the lowest `e_final` among the first N candidates. Efficiency = (random − picked) / (random − oracle).

- **No diversity, no verification.** Without Langevin noise, the loop forgets its initial noise (Q3: it converges to nearly the same fixed point), so all 32 candidates are almost identical, and the best possible gain (oracle) is under 0.005 nats.
- **With diversity, the energy picks the wrong candidates.**
  - With σ = 3, there is a lot to gain: the oracle-32 is 0.51 nats better than a single sample.
  - Energy selection captures that only at N = 2 (−0.010 nats) and becomes harmful from N = 4 on, which is the "adversarial sample" effect the paper reports for early checkpoints (Fig. B.1a).
  - Picking the *highest* energy is also worse than random (4.0933 at N = 32). So CE is U-shaped in energy, and the middle of the candidate set is best.

### 2.2 Ranking consistency and what the energy tracks

Share of same-token candidate pairs where the lower-energy candidate is "better":

| Run | Lower E ⇒ lower CE (any gap) | Lower E ⇒ lower CE (\|ΔCE\| > 0.1) | Same, with `e_last_logged` | Lower E ⇒ lower entropy |
|---|---|---|---|---|
| S1 105k, σ=0 | 0.525 | 0.389 | 0.366 | 0.727 |
| S2 105k, σ=0 | 0.555 | 0.304 | 0.462 | 0.736 |
| **S2 105k, σ=3** | **0.634** | **0.623** | 0.389 | **0.982** |
| S2 90k, σ=3 | 0.612 | 0.592 | 0.419 | 0.981 |

- With real candidate diversity (σ = 3), energy ranks by correctness at 0.62–0.63, above chance but below VFScale's ≈0.73. It ranks by **sharpness** at 0.98.
- Mechanism (our interpretation): the energy is mostly a sharpness detector. Lower-energy candidates are more confident. When confidence lands on the right token, CE improves; otherwise it gets much worse. Larger N finds sharper candidates, which drives BoN CE up.
- In the σ = 0 runs, the |ΔCE| > 0.1 column covers few pairs, because candidates are nearly identical.
- The energy that training logs (`e_last_logged`, before the final update) is *anti*-consistent (0.37–0.46). It must not be used as a verifier.

### 2.3 Off-trajectory candidates (σ = 0 runs and S2 σ = 3 agree; S2 105k σ = 3 shown)

| Candidate | Mean energy | Mean CE | Share of tokens where it has the lowest energy (of 7) |
|---|---|---|---|
| EBT's own prediction | 3.169 | 3.92 | 2.7% |
| Transformer++, T = 0.5 | 2.929 | 5.02 | 22.4% |
| Transformer++, T = 1 | 3.186 | 3.67 | 0.4% |
| Transformer++, T = 2 | 5.198 | 5.09 | 0% |
| one-hot(target) | 3.177 | 0 | 30.3% |
| one-hot(Transformer++ argmax) | 2.909 | 17.6 | **44.2%** |
| uniform | 9.229 | 10.83 | 0% |

- The energy prefers sharp distributions regardless of whether they are right. One-hot of a usually-wrong token (Transformer++'s top-1 is wrong for ≈64% of tokens, derived from that row's mean CE and the 1e-12 floor) beats the true one-hot most often. Sharpening Transformer++ (T = 0.5) lowers energy while raising CE by 1.35 nats.
- Some correctness signal does exist: one-hot(target) has lower energy than EBT's own prediction for 64% (S2) and 76% (S1) of tokens.
- Transformer++'s T = 1 distribution, which has lower CE on average (3.67 vs 3.92), gets lower energy than EBT's own prediction in 45% (S2 σ = 3) and 72% (S2 σ = 0) of tokens. So the energy can score distributions it never produced, but not reliably in the direction of quality.
- S1 shows the same pattern on its much smaller energy scale (mean energies 0.149–0.179; one-hot argmax lowest for 53%).

### 2.4 Energy as an uncertainty signal (across tokens)

| Run | AUROC(energy → top-1 error) | AUROC(entropy → top-1 error) | Spearman(energy, CE) | Spearman(entropy, CE) |
|---|---|---|---|---|
| S1 105k | 0.677 | 0.824 | 0.355 | 0.687 |
| S2 105k, σ=0 | 0.623 | 0.808 | 0.223 | 0.714 |
| S2 105k, σ=3 | 0.628 | 0.806 | 0.233 | 0.675 |

Energy does carry difficulty information (AUROC > 0.5; the paper's "Facet 2"), but the model's own predictive entropy is a clearly better uncertainty signal at no extra cost.

## 3. Implications

- **For Q1 (thinking vs Transformer++):** at xxs/105k, best-of-N by energy cannot close the gap to Transformer++ (3.67 on this subset); with σ = 3 it moves *away* from it. The "+self-verification" arm of the paper's thinking claims fails in-distribution here. OOD remains untested.
- **For Gap 2 in [[LIT_REVIEW]]:** independent evidence that an unrolled-optimization-trained energy is a weak, sharpness-biased verifier. It is consistent with VFScale's finding (≈0.73) and with Fig. B.1a of the EBT paper, and here it is even weaker.
- **Training-time hypothesis** (not tested): the energy is never trained to rank candidates. The loss only supervises the final prediction (truncated), and `contrastive_loss` is off. A ranking or contrastive term on candidates of equal sharpness would directly target the failure seen here.

> [!warning] Scope
> - xxs, one seed, in-distribution validation only, 228K tokens.
> - Candidates come from initial noise and Langevin noise at the trained step count. Other candidate sources (temperatures, step counts) could rank differently.
> - The 90k → 105k change is two checkpoints from one run, not a trend.

## 4. Reproduction

```bash
cd refs/EBT/experiments/thinking_study
GPU_ID=0 bash q2_verifier_quality/run_all.sh        # 5 configs, ~30 min on one H100 NVL
PYTHONPATH=$PWD/common /mnt/data/conda-envs/quangnx/ebt312/bin/python q2_verifier_quality/analyze_q2.py q2_verifier_quality/results/*
```
