---
title: "Energy momentum — implementation plan"
date: 2026-10-29
tags: [ebt, momentum, inference, thinking-study, plan]
status: plan
---

# Energy Momentum Implementation Plan

> [!summary] TL;DR
> - Replace plain gradient descent in the EBT inference loop with **momentum** (heavy-ball and a bias-corrected EMA), training-free, on the final `s2_paper` checkpoint (step 1M, 128M sequences). Run up to 32 steps per trajectory, so one run gives CE at every k (the trained length k = 3 and Extension 1's k = 5, 8, 10).
> - A **step-size control** (plain GD with larger α) separates "momentum helps" from "a bigger effective step helps". Q3 already found the CE-optimal step along the update to be ≈ 1.75 × α. Six pre-registered claims (M1–M6), compared with T++ and with EBT-GD, ours and the paper's.
> - **Extension 2 (retraining with momentum) is gated.** It runs only if the training-free claims fail and you approve the compute (≈ 3.5 GPU-days at the 16M-sequence scale). Phases 1–2 cost ≈ 4 GPU-hours.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether momentum in the energy-minimization loop improves S2's perplexity at the trained step count, makes extra steps ("thinking longer") useful, and makes self-verification (BoN) useful.

**Architecture:**
- A new generator `momentum_mcmc` mirrors `harness.mcmc` (`../../thinking_study/common/harness.py:154`). It draws random numbers in the same order, so every arm sees the same noise (common random numbers) and differs only in its update rule.
- `run_momentum.py` records per-step arrays, using Q3 Part A field names so that `analyze_q3_extra.cycle_metrics` works unchanged. `run_bon.py` records BoN candidates. `collect_momentum.py` prints the verdicts.
- Phase 3, if approved, adds the same rule to `model/nlp/ebt.py` behind a flag whose default keeps current behavior bit-identical.

**Tech Stack:** PyTorch 2.4 + Lightning 2.4, numpy, scipy, matplotlib, pytest; env `/home/quangnx/miniconda3/envs/ebt312`.

**Spec:** [[ENERGY_MOMENTUM]] (`../ENERGY_MOMENTUM.md`), plus the decisions in §0 below. Context: [[report|paper-scale report]] (`../report/report.md`, §2.5 for the landscape findings) and [[SUMMARIZED_FINDINGS]].

## 0. Spec: decisions and rationale

### 0.1 Update rules

The repo's loop (`model/nlp/ebt.py:115-171`, `harness.mcmc`) is:
- `p_t = softmax(z_t + η_t)` with `η_t ~ N(0, σ²)`;
- `g_t = ∂E/∂p` at `p_t`;
- `z_{t+1} = p_t − α g_t`.

Each step re-anchors at `p_t`. Momentum can therefore only act through the gradient term:

| Rule | Update | Why |
|---|---|---|
| `gd` | `z_{t+1} = p_t − α g_t` | Baseline; bit-identical to the repo |
| `hb` (heavy-ball) | `v_t = μ v_{t−1} + g_t`, `z_{t+1} = p_t − α v_t`, `v_{−1} = 0` | Classic momentum, as the spec asks. Its steady-state step is α/(1−μ), so any gain is confounded with step size |
| `ema` (bias-corrected EMA) | `m_t = μ m_{t−1} + (1−μ) g_t`, `z_{t+1} = p_t − α m_t / (1 − μ^{t+1})` | Step-size neutral: equals `gd` when the gradient is constant. It isolates direction averaging, which should damp the even/odd 2-cycles found in Q3 (29.9% of tokens at t = 31) |

- Nesterov is **not** run. It needs the gradient at a look-ahead point, which changes where the energy is measured. Add it only if `hb` or `ema` shows a gain (YAGNI).
- Softmax ignores a constant shift across the vocabulary, so `g` is not centred.

### 0.2 Arms and grid

| Arm name | Rule | μ | α |
|---|---|---|---|
| `gd_a5` | gd | 0 | 5 (trained; **baseline**) |
| `gd_a7.5`, `gd_a10`, `gd_a15`, `gd_a20` | gd | 0 | 7.5, 10, 15, 20 (**step-size controls**; `hb` steady state at μ = 0.5 / 0.7 / 0.9 is 10 / 16.7 / 50) |
| `hb_m0.3`, `hb_m0.5`, `hb_m0.7`, `hb_m0.9` | hb | 0.3 … 0.9 | 5 |
| `ema_m0.3`, `ema_m0.5`, `ema_m0.7`, `ema_m0.9` | ema | 0.3 … 0.9 | 5 |

- **13 arms × 5 sets** (rp_val, GSM8K, SQuAD, BB Elementary Math QA, BB Dyck) × σ ∈ {0, 3} × 32 steps, on the **same 1,024 sequences** as `../results/q3` (`eval_subset_indices(n, 1024)`, batch 16, `torch.manual_seed(seed + b)`, seed 0).
- Cost: ≈ 3 min per set per arm, so ≈ 2.7 GPU-hours, ≈ 45 min on 4 GPUs.
- **Model:** `s2_paper` at step 1,000,000. T++ reference: `tpp_paper` at step 1,000,000.
- **Selection without peeking at OOD:**
  - `arm*` is the momentum arm with the lowest rp_val CE at k = 3, σ = 3.
  - `gd*` is the GD control (all `gd_*` arms) chosen by the same rule.
  - `k*` is the k ∈ {5, 8, 10} with the lowest rp_val CE for `arm*` at σ = 3.
  - OOD sets are reported only for these choices; the full grid goes to an appendix table.
- **Self-verification (BoN):** run only for `gd_a5` and `arm*`. 10 candidates at k = 3 (and at `k*` if `k* ≠ 3`), σ = 3. Pick the lowest final energy on the last landscape, as in Q2. Seeds: `seed * 1_000_003 + b * 1009 + g`.

### 0.3 Definitions

- **CE after k steps** = mean over tokens of −log p(target) under `softmax(z_k)`. In the arrays this is `logp[k]` for k < N and `logp_final` for k = N (the Q3 Part A convention).
- **ppl**, **% ppl improvement** and **mean OOD ppl** (the arithmetic mean of the 4 per-set ppl) are as in `../IMPLEMENTATION_PLAN.md` §0.1. The geometric mean is reported as a secondary row.
- **Paired CI:** a 95% paired bootstrap over sequences (1,000 resamples, numpy seed 0) of Δ mean CE (arm − `gd_a5`). Arms share noise draws, so the comparison is paired.
- **2-cycle share:** `analyze_q3_extra.cycle_metrics(...)["share_two_cycle"]` on rp_val, σ = 0, N = 32.

### 0.4 Pre-registered claims

These are fixed before any momentum run. `collect_momentum.py` prints each verdict.

| # | Claim | Supported if | Reference (GD, ours) |
|---|---|---|---|
| M1 | Momentum helps at the trained length | `arm*` at k = 3, σ = 3: mean OOD ppl lower than `gd_a5` by **> 2%**, **and** the paired CI of ΔCE is below 0 on **≥ 3 of 4** OOD sets | k = 3: 82.76 |
| M2 | The gain is not just step size | `arm*` mean OOD ppl at k = 3 is lower than `gd*` by **> 1%** | – |
| M3 | Momentum makes thinking longer useful (Ext. 1) | `arm*` mean OOD ppl at `k*` is lower than its own k = 3 by **> 2%** | GD: k = 32 is +0.106 nats worse than k = 3 |
| M4 | Momentum damps the late-loop 2-cycles | rp_val 2-cycle share for `arm*` **< 15%** (half of GD) | 29.9% |
| M5 | Momentum makes self-verification useful | `arm*` BoN-10 vs 1 sample (k = 3, σ = 3): mean OOD % ppl improvement **> 0** | −15.3% (BoN-10 vs k = 3) |
| M6 | Momentum closes the downstream gap to T++ | `arm*` (k = 3, σ = 3) ppl < T++ on **≥ 1 of 4** OOD sets | 0 of 4 |

- **Prediction, written down so it can be checked:** M5 most likely fails. Momentum changes the path, not the energy, and the energy ranks candidates almost entirely by entropy (R² 0.973, `../report/report.md` §2.5).
- **Gate to Phase 3:** run Phase 3 if M1 **and** M3 are both Not supported, and only after explicit approval from the user (compute).

### 0.5 Baselines in the report

| Row | Source |
|---|---|
| T++ (ours) | `tpp_ce` in `../results/q2/s2_paper_step1000000_<set>_noise3/metrics.json` |
| EBT-GD (ours), k = 2 / 3 / BoN-10 | `gd_a5` arm (recomputed, paired), checked against `../report/result.md` |
| T++ / EBT (paper) | Table 3: T++ [31.36, 49.6, 52.3, 79.8, 131.5], EBT [33.43, 43.3, 53.1, 72.6, 125.3] (rp_val, GSM8K, SQuAD, BB math, Dyck) |
| Paper Fig. 6a | Relative only: k = 2 → 3 ≈ 20%, k = 3 → BoN-10 ≈ 11% |

## Global Constraints

- Python: `/home/quangnx/miniconda3/envs/ebt312/bin/python`; `PYTHONPATH="$REPO:$REPO/experiments/thinking_study/common"`.
- **Only GPUs 0, 1, 2, 3.** Launch scripts source `../common.sh` and call `require_gpus`, which refuses GPUs 4 and 5, a busy GPU or < 500 GB free disk.
- Workspace `WS = code/EBT/experiments/scale_to_paper_level/energy_momentum/`. Outputs go to `$WS/results/` and `$WS/report/`. `results/` is gitignored; `report/` is committed.
- Branch `exp/energy-momentum` from `scale-paper` HEAD. Commit per task; **never push**.
- Do not modify `../results/`, `../report/` or `../../thinking_study/`; import from them only. Phase 3 changes to `model/nlp/ebt.py` and `train_model.py` must keep `tests/` passing.
- Every file in English. Markdown is Obsidian-friendly with a 3-bullet TL;DR. Python follows `python-guidelines-research`.

## Review Focus

1. **The `gd_a5` arm must reproduce `../results/q3` bit-for-bit (or within 1e-5).** Otherwise every paired comparison is off by a different noise draw. Task 2 tests this against the stored `part_a_noise3.npz`.
2. **Bias correction at t = 0.** `ema` must equal `gd` exactly on the first step for any μ (`1 − μ^1 = 1 − μ`). An off-by-one in the exponent silently shrinks the first step. This is pinned in Task 1.
3. **Momentum state crossing batches or candidates.** The velocity must start at 0 for every trajectory (every batch, every BoN candidate). This is pinned in Task 1 by running the generator twice.
4. **Dyck has only 10,818 tokens.** Its per-set CI will often straddle 0, so M1 needs only 3 of 4 sets. `collect_momentum.py` must print Dyck's CI and not drop it.
5. **Phase 3: `--mcmc_momentum 0` must leave training bit-identical**, including the replay buffer path. The replay buffer stores logits only, so a replayed trajectory restarts with v = 0. This is documented, not fixed (Task 6).

---

## Phase 1–2: training-free

### Task 1: Momentum inference loop

**Files:**
- Create: `$WS/momentum.py`
- Test: `$WS/test_momentum.py`

**Interfaces:**
- Consumes: `harness.Step`, `harness.energy`, `harness.landscape_index`.
- Produces:
  - `Arm(name: str, rule: Literal["gd", "hb", "ema"], mu: float = 0.0, alpha: float = 5.0)`, a frozen dataclass.
  - `ARMS: dict[str, Arm]`, the 13 arms of §0.2, keyed by name.
  - `momentum_mcmc(model, input_ids, n_steps: int, noise_std: float, arm: Arm, z0=None) -> Generator[Step, None, torch.Tensor]`, the same contract as `harness.mcmc`.
  - `run_momentum_mcmc(model, input_ids, n_steps, noise_std, arm, on_step=None) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests** with a CPU stub model.
  - The stub is a `SimpleNamespace` with `embeddings = nn.Embedding(11, 4)`, `vocab_to_embed = nn.Linear(7, 4)`, a `transformer(x, start_pos, mcmc_step)` that returns `(x[:, S:] @ w).unsqueeze(-1)` with a nonlinearity (`tanh`) so the gradient is not constant, `corrupt_embeddings = lambda e: torch.randn(B, S, 7)`, `alpha = torch.tensor(5.0)` and `hparams.mcmc_num_steps = 1`.
  - For the constant-gradient tests, use a variant without `tanh`.

```python
def test_gd_matches_harness_mcmc():            # same seed -> identical z_N, energies, grads; σ = 3, N = 6
def test_hb_and_ema_with_mu0_equal_gd():       # torch.equal on z_N
def test_ema_equals_gd_for_constant_gradient():  # linear stub, μ = 0.7: allclose(z_N, gd z_N, atol=1e-6)
def test_hb_constant_gradient_step_sizes():    # linear stub, σ = 0: z_{t+1} - p_t == -α g (1 - μ^{t+1}) / (1 - μ)
def test_ema_first_step_equals_gd():           # nonlinear stub, any μ: z_1 identical
def test_velocity_resets_per_call():           # two calls with the same seed give identical z_N
def test_arms_registry():                      # 13 names; ARMS["gd_a5"] == Arm("gd_a5", "gd", 0.0, 5.0)
```

- [ ] **Step 2: Run the tests and check that they fail.** Run `$PY -m pytest $WS/test_momentum.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `momentum_mcmc`.**
  - Copy the structure of `harness.mcmc`: initial `corrupt_embeddings`, then one Langevin draw per step, in this order.
  - Keep `v` as a local tensor initialised to `torch.zeros_like(g)` at t = 0.
  - Use `arm.alpha` and ignore `model.alpha`.
  - Yield `Step` before the update, exactly as `harness.mcmc`.
- [ ] **Step 4: Run the tests and check that they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat(momentum): heavy-ball and bias-corrected EMA EBT inference loop"`

### Task 2: Per-step runner and the baseline reproduction check

**Files:**
- Create: `$WS/run_momentum.py`, `$WS/run_all.sh`, `$WS/.gitignore` (`results/`)
- Test: `$WS/test_run_momentum.py`

**Interfaces:**
- Consumes: `ARMS`, `momentum_mcmc`; `harness.load_model`, `eval_dataset`, `eval_subset_indices`, `batches`, `token_logprob`, `entropy`.
- Produces:
  - `results/single/<arm>_<set>/part_a_noise{0,3}.npz`. Fields: `energy`, `entropy`, `logp`, `tv_next` (each of shape (N, tokens)), `logp_final` (tokens), `seq_id` (tokens; the position in `idx`).
  - `config.json` (args + `alpha`, `mu`, `rule`, `seconds`).
  - The CLI: `run_momentum.py --arm NAME --data SET [--model s2_paper --step 1000000 --n-seqs 1024 --n-steps 32 --batch-size 16 --seed 0] --out-dir DIR`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_record_fields_match_q3(tmp_path):   # run on the CPU stub with 2 batches; npz keys ⊇ {energy, entropy, logp, tv_next, logp_final, seq_id}
                                             # shapes (N, T) / (T,); seq_id is non-decreasing
def test_cycle_metrics_accepts_output(tmp_path):  # analyze_q3_extra.cycle_metrics(np.load(...)) returns share_two_cycle in [0, 1]
@pytest.mark.skipif(not torch.cuda.is_available() or not CKPT.exists(), reason="needs GPU and s2_paper ckpt")
def test_gd_a5_reproduces_q3_part_a():        # first 64 seqs of rp_val, σ = 3: logp allclose(atol=1e-5) to
                                              # ../results/q3/s2_paper_step1000000_rp_val/part_a_noise3.npz["logp"][:, :T64]
```

- [ ] **Step 2: Run them and check that they fail.**
- [ ] **Step 3: Implement `record(model, ds, idx, n_steps, noise, arm, batch_size, seed) -> dict[str, np.ndarray]` and `main()`.**
  - Per batch b, call `torch.manual_seed(seed + b)`, as Q3 `part_a`.
  - Mask with the collate mask.
  - Compute `tv_next` as in `run_q3.part_a`, including the final step's TV against `softmax(z_N)`.
- [ ] **Step 4: Write `run_all.sh`.**
  - It sources `../common.sh` and calls `require_gpus "$GPU_ID"`.
  - It loops `ARMS` × `DATASETS` (default: all 5).
  - It skips an output whose `config.json` exists.
  - Usage: `GPU_ID=0 ARMS="gd_a5 gd_a7.5 gd_a10" bash run_all.sh`.
- [ ] **Step 5: Run the tests on GPU 0 and check that they pass,** including the reproduction test.
- [ ] **Step 6: Commit.** `git commit -m "feat(momentum): per-step runner reproducing Q3 Part A for gd_a5"`

### Task 3: Run the grid

- [ ] **Step 1:** Launch 4 shards in parallel, one tmux session each:
  - GPU 0: `gd_a5 gd_a7.5 gd_a10 gd_a15`
  - GPU 1: `gd_a20 hb_m0.3 hb_m0.5`
  - GPU 2: `hb_m0.7 hb_m0.9 ema_m0.3`
  - GPU 3: `ema_m0.5 ema_m0.7 ema_m0.9`
  - Logs go to `$WS/logs/grid_g{0..3}.log`.
- [ ] **Step 2: Check.** There should be 65 `config.json` files under `results/single/`, and no NaN in any `logp`: `$PY -c "import numpy as np,glob; assert all(np.isfinite(np.load(f)['logp']).all() for f in glob.glob('results/single/*/*.npz'))"`.
- [ ] **Step 3: Sanity check.** `gd_a5` CE at k = 2 and k = 3 must match `../report/result.md` (Fig. 6a rows) to 4 decimals in mean OOD ppl.

### Task 4: Self-verification runner

**Files:**
- Create: `$WS/run_bon.py`
- Test: `$WS/test_run_bon.py`

**Interfaces:**
- Consumes: `ARMS`, `run_momentum_mcmc`, `harness.energy`.
- Produces: `results/bon/<arm>_k<k>_<set>/arrays.npz` with `e_final` and `logp` (shape (G, tokens)) and `seq_id` (tokens), plus `config.json`. The CLI is `run_bon.py --arm NAME --k K --data SET [--n-candidates 10 --noise 3 --n-seqs 1024] --out-dir DIR`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_bon_shapes_and_seeds(tmp_path):       # CPU stub, G = 3: shapes (3, T); candidate 0 equals a single
                                               # run_momentum_mcmc call seeded with seed*1_000_003 + 0*1009 + 0
def test_energy_on_last_landscape(tmp_path):   # e_final == harness.energy(model, ctx, softmax(z_N), mcmc_num_steps - 1)
```

- [ ] **Step 2: Run them, check that they fail, implement, then check that they pass.** The structure follows `run_q2.py` minus the off-trajectory and diversity parts.
- [ ] **Step 3: Run after Task 5 selects `arm*` and `k*`.** Use `gd_a5` and `arm*` at k = 3, plus k = `k*` if it differs, over all 5 sets, on GPUs 0–3.
- [ ] **Step 4: Commit.** `git commit -m "feat(momentum): BoN self-verification runner"`

### Task 5: Collector, verdicts and report

**Files:**
- Create: `$WS/collect_momentum.py`, `$WS/report/result.md` (generated), `$WS/report/figures/*.png`, `$WS/report/report.md` (written by hand)
- Test: `$WS/test_collect_momentum.py`

**Interfaces:**
- Consumes: Task 2 and Task 4 outputs; `analyze_q3_extra.cycle_metrics`; `../collect_paper.pct_improvement`; `../results/q2/*/metrics.json` for T++ CE.
- Produces:
  - `ce_after_k(a: dict, k: int) -> np.ndarray` (per-token CE);
  - `mean_ood_ppl(ce_by_set: dict[str, float], geometric: bool = False) -> float`;
  - `paired_bootstrap(ce_a, ce_b, seq_id, n=1000, seed=0) -> tuple[float, float, float]` (mean Δ, lo, hi);
  - `select(results) -> tuple[str, str, int]` (`arm*`, `gd*`, `k*`);
  - `verdicts(...) -> dict[str, str]` for M1–M6.

- [ ] **Step 1: Write the failing tests with synthetic arrays.**

```python
def test_ce_after_k_convention():      # k < N -> -logp[k]; k == N -> -logp_final
def test_mean_ood_ppl():               # {a: log 10, b: log 30} -> 20.0; geometric -> sqrt(300)
def test_paired_bootstrap_sign():      # ce_a = ce_b - 0.1 on every token -> hi < 0
def test_selection_uses_rp_val_only(): # an arm best on OOD but not on rp_val is not chosen
def test_m1_needs_three_sets():        # Δppl 3% but CI < 0 on only 2 of 4 sets -> "Not supported"
def test_gate_flag():                  # M1 and M3 both not supported -> phase3_gate == True
```

- [ ] **Step 2: Run them, check that they fail, implement, then check that they pass.**
- [ ] **Step 3: Generate `result.md`.** It contains:
  - the verdict table (M1–M6);
  - the baselines table (§0.5) with `arm*` and `gd*` added;
  - CE vs k for k ∈ {1…10, 16, 32} per arm on rp_val and mean OOD;
  - 2-cycle shares;
  - the full-grid appendix;
  - the `phase3_gate` line.
- [ ] **Step 4: Generate the figures.**
  - `ce_vs_k.png`: mean OOD ppl vs k, with lines for `gd_a5`, `gd*`, `arm*` and dashed T++.
  - `grid_heatmap.png`: rp_val CE at k = 3, arms × σ.
  - `bon.png`: BoN-N vs N for `gd_a5` and `arm*`.
  - Load the `dataviz` skill before writing the plotting code.
- [ ] **Step 5: Write `report/report.md`.** Use a 3-bullet TL;DR and the structure of `../report/report.md`: Setup, Results per claim, Verdicts, Interpretation, Limitations, Reproduction.
- [ ] **Step 6: Commit.** `git commit -m "feat(momentum): collector, verdicts M1-M6 and report"`
- [ ] **Step 7: Stop and report to the user.** Give the verdicts and the gate. Do not start Phase 3 without approval.

## Phase 3: retrain with momentum (gated, needs approval)

> [!warning] Run only if `phase3_gate` is true **and** the user approves
> Cost ≈ 3.5 days on GPUs 0–3. The scale is the 16M-sequence recipe (`../../scale_to_16m_sequences/`: global batch 64 = 2 × 32, 250k steps), not 128M: each S2 run at 128M takes 90 h. The baseline is the existing `s2_b64_16m` (GD, 2–3 steps).

| Run | Rule, μ | Train steps | Purpose |
|---|---|---|---|
| `s2-mom-16m` | `arm*` rule and μ | 2–3 (as baseline) | Does training with momentum help? |
| `s2-mom-long-16m` | `arm*` rule and μ | 4–6 (`--randomize_mcmc_num_steps 5 --randomize_mcmc_num_steps_min 4`) | Spec: "more steps and momentum" |
| `s2-gd-long-16m` | gd | 4–6 | Control: more steps without momentum |

- Each run uses 2 GPUs × 32, which matches the baseline's per-device batch and therefore its replay-buffer sample size.
- Schedule: `s2-mom-16m` (GPUs 0,1, ≈ 23 h) runs in parallel with `s2-mom-long-16m` (GPUs 2,3, ≈ 40 h); then `s2-gd-long-16m` runs.

### Task 6: Momentum in the training loop

**Files:**
- Modify: `model/nlp/ebt.py:164-171` (the update), `train_model.py` (two arguments next to `--scale_alpha_with_energy`, line 401)
- Test: `tests/test_ebt_momentum.py`

**Interfaces:**
- Produces:
  - `--mcmc_momentum` (float, default 0.0);
  - `--mcmc_momentum_rule` (`hb` | `ema`, default `hb`);
  - the same math as §0.1, applied to `predicted_tokens_grad`.
- With `truncate_mcmc`, gradients from non-final steps carry no graph, so `v_{t−1}` is detached and only `g_last` carries the graph.
- Raise `ValueError` if this is combined with `--scale_alpha_with_energy`.

- [ ] **Step 1: Write the failing tests.** Use a tiny `EBT_NLP` on CPU (xxs config with 2 layers and vocab from the test tokenizer, as in `tests/test_param_counts.py`).

```python
def test_momentum_zero_is_bit_identical():   # same seed, train-mode forward with flags absent vs --mcmc_momentum 0: torch.equal on all outputs and on loss.backward() grads
def test_momentum_matches_inference_loop():  # eval-mode forward (no_randomness, σ = 0) with ema μ = 0.5 == momentum_mcmc(..., ARMS-like Arm("x", "ema", 0.5, 5.0)) final logits
def test_momentum_rejects_scaled_alpha():    # ValueError
```

- [ ] **Step 2: Run them, check that they fail, implement, then check that they pass.** Run the full `pytest tests/` suite as well.
- [ ] **Step 3:** Add `"s2_mom_16m"`, `"s2_mom_long_16m"` and `"s2_gd_long_16m"` to `harness.RUN_DIRS`. This is the only edit to `thinking_study/`; it adds keys and changes no behavior.
- [ ] **Step 4: Commit.** `git commit -m "feat(ebt): optional momentum in the MCMC update (default off, bit-identical)"`

### Task 7: Smoke test, the three runs, evaluation

**Files:**
- Create: `$WS/run_retrain.sh`. It wraps `../../scale_to_16m_sequences/run_s2.sh`, but enforces `require_gpus` from `../common.sh`, sets `--wandb_project ebt_momentum`, and passes the extra flags through.

- [ ] **Step 1: Smoke test.** Run `--max_steps 200 --val_check_interval 100` for `s2-mom-long-16m` on GPUs 0,1. Expected: no NaN or Inf, and validation runs at steps 100 and 200.
- [ ] **Step 2: Launch the three runs** per the schedule, under `watch_run.sh` (`../watch_run.sh`).
- [ ] **Step 3: Evaluate each final checkpoint.**
  - Run `run_momentum.py` with the arm matching its training rule (`gd_a5` for `s2-gd-long-16m`) on all 5 sets.
  - Run `run_bon.py` at the trained k.
  - Compare with `s2_b64_16m` under `gd_a5` and under the training-free `arm*` (paired, same seeds).
- [ ] **Step 4:** Extend `collect_momentum.py` with a retrain section of claims.
  - **R1:** `s2-mom-16m` validation CE is lower than `s2_b64_16m` by > 0.01 nats.
  - **R2:** `s2-mom-long-16m` mean OOD ppl at its trained k is lower than `s2-gd-long-16m` by > 2%.
  - **R3:** BoN-10 > 1 sample for any retrained model.
  - Add tests in the Task 5 style first.
- [ ] **Step 5:** Update `report/report.md` with a Phase 3 section, then commit.
