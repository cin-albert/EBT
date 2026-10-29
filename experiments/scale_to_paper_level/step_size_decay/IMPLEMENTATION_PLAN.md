# Step-size decay in EBT inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (Native). Steps use `- [ ]` checkboxes.

> [!summary] TL;DR
> - **Question:** does EMA momentum's 12.7% OOD gain at k = 3 (`../energy_momentum/report/report.md`) come from averaging gradient directions, or just from taking smaller steps after the first one? Plain GD with a decaying or smaller α is the missing control (that report's M2 caveat, next step 1).
> - **Design:** 8 training-free arms on the same checkpoint, the same 1,024 sequences per set and the same seeds as the momentum grid, so every arm is paired with `gd_a5` and `ema_m0.5`. Four GPUs (0–3), ≈ 30 min.
> - **Decision rule fixed in advance (D1):** the best decay arm, selected on rp_val, at ≤ 73.2 mean OOD ppl means "step size explains most of the EMA gain"; at ≥ 78.0 it means "momentum does more than shrink the step"; in between, both contribute.

**Goal:** test whether a GD step-size schedule matches bias-corrected EMA momentum at the trained step count.

**Architecture:** extend the shared `energy_momentum/momentum.py` `Arm` with an α schedule (default constant, so all existing arms are unchanged and stay bit-exact). New arms, launcher, collector and report live in `step_size_decay/`. Recording reuses `energy_momentum/run_momentum.record` unchanged.

**Tech stack:** PyTorch, numpy, matplotlib; env `/home/quangnx/miniconda3/envs/ebt312`.

**Spec:** this file (§0) and `../energy_momentum/report/report.md` §2.3, §4.1.

## 0. Design

### 0.1 Rules (inherited from the momentum plan)
- Checkpoint `s2_paper` step 1M; k up to 32; σ ∈ {0, 3}; arms recorded per step, so any k ≤ 32 is read from one run.
- Same 1,024 sequences per set (`eval_subset_indices`, seed 0), batch 16, `torch.manual_seed(seed + b)`: common random numbers across all arms, old and new.
- Langevin noise is **not** decayed: σ is added to the logits each step regardless of α. This is the repo's inference loop; decaying σ too is out of scope.
- GPUs 0–3 only (`require_gpus`). Commit locally, never push.

### 0.2 Schedules and arms

α_t is the step size of update t = 0, 1, … (update 0 leaves the random initial state).

| Schedule | α_t |
|---|---|
| `const` | α₀ |
| `harmonic` | α₀ / (t + 1) |
| `geom` | α₀ · γ^t |

| Arm | Rule | α₀ | Schedule | α₀, α₁, α₂ (k = 3) | Purpose |
|---|---|---|---|---|---|
| `gd_a2.5` | gd | 2.5 | const | 2.5, 2.5, 2.5 | small constant step |
| `gd_a3.5` | gd | 3.5 | const | 3.5, 3.5, 3.5 | small constant step |
| `gd_a5_harm` | gd | 5 | harmonic | 5, 2.5, 1.67 | trained first step, then decay |
| `gd_a5_g0.5` | gd | 5 | geom 0.5 | 5, 2.5, 1.25 | faster decay |
| `gd_a5_g0.7` | gd | 5 | geom 0.7 | 5, 3.5, 2.45 | slower decay |
| `gd_a7.5_harm` | gd | 7.5 | harmonic | 7.5, 3.75, 2.5 | large first step (gd_a7.5 is the best GD at k = 2) |
| `gd_a7.5_g0.7` | gd | 7.5 | geom 0.7 | 7.5, 5.25, 3.68 | large first step, slow decay |
| `ema_m0.5_harm` | ema | 5 | harmonic | – | exploratory: are the two levers additive? |

Reused from `../energy_momentum/results/single/`: `gd_a5` (baseline), `ema_m0.5` (reference), `gd_a7.5`.

### 0.3 Pre-registered claims

`d*` = the arm with the lowest rp_val CE at k = 3, σ = 3 among the 7 GD arms of §0.2 (not `ema_m0.5_harm`). The headline metric is the arithmetic mean of the 4 OOD sets' ppl at k = 3, σ = 3, as in the momentum report. References: `gd_a5` 82.76, `ema_m0.5` 72.21, best GD with α ≥ 5 over all k 78.0.

| # | Claim | Supported if |
|---|---|---|
| D1 | A step-size schedule explains the EMA gain | `d*` mean OOD ppl ≤ 73.2 → "step size explains most"; ≥ 78.0 → "momentum does more" (D1 not supported); else "partial" |
| D2 | Decay beats the trained constant step | `d*` vs `gd_a5`: > 2% mean OOD ppl improvement and paired 95% CI < 0 on ≥ 3 of 4 OOD sets |
| D3 | Decay removes the 2-cycle | share of rp_val tokens in a 2-cycle at t = 31, σ = 0, for `d*` < 5% (`gd_a5`: 29.6%) |
| D4 | Decay stops the drift with k | `d*` mean OOD ppl at k = 32 is < 2% above its value at k = 3 (`ema_m0.5`: +8.6%) |
| D5 | (exploratory) Momentum and decay add up | `ema_m0.5_harm` vs `ema_m0.5` at k = 3: paired CI < 0 on ≥ 3 of 4 OOD sets |

Every table also shows `d*` vs `ema_m0.5` with paired CIs per set. Results on OOD arms other than `d*` are reported but marked post hoc.

### 0.4 Review focus
- **Constant-schedule arms must be bit-identical to the old ones:** a default-schedule `Arm` must give the same z as before for gd, hb and ema.
- **Off-by-one in t:** update 0 must use α₀ (harmonic and geom both give α₀ at t = 0).
- **Paired comparisons need identical tokens:** assert equal `seq_id` before bootstrapping across workspaces.

## Task 1: α schedule in the shared `Arm`

**Files:** modify `../energy_momentum/momentum.py`, `../energy_momentum/test_momentum.py`.

**Produces:** `Arm(..., schedule: Literal["const", "harmonic", "geom"] = "const", gamma: float = 1.0)`; `step_size(arm: Arm, t: int) -> float`; `momentum_mcmc` uses `step_size(arm, t)` in place of `arm.alpha`.

- [ ] Write the failing tests: `test_step_size_schedules` (const 5,5,5; harmonic 5, 2.5, 5/3; geom 0.5 → 5, 2.5, 1.25) and `test_decay_scales_each_update` (stub model, linear energy so g is constant: z_{t+1} − p_t = −α_t g for gd with harmonic).
- [ ] Run, expect FAIL (no `step_size`).
- [ ] Implement.
- [ ] Run the whole suite: the existing bit-exact test of gd against `harness.mcmc` must still pass; also run the GPU test against `../results/q3`.
- [ ] Commit.

## Task 2: arms, recorder CLI and launcher

**Files:** create `decay_arms.py`, `run_decay.py`, `run_all.sh`, `conftest.py`, `test_decay_arms.py`.

**Consumes:** `Arm`, `run_momentum.record`, `save_record`. **Produces:** `DECAY_ARMS: dict[str, Arm]` (§0.2); `run_decay.py --arm --data --out-dir` writing `results/single/<arm>_<set>/{part_a_noise0,part_a_noise3}.npz, config.json`, the same layout as the momentum grid.

- [ ] Write the failing test `test_registry` (8 names of §0.2; α₀..α₂ per arm match the table).
- [ ] Run, expect FAIL; implement; run, expect PASS.
- [ ] Smoke on GPU: `gd_a5`-equivalent `Arm("x", "gd", 0, 5, "const")` through `run_decay.record` on 64 rp_val sequences equals `../energy_momentum/results/single/gd_a5_rp_val` on those sequences.
- [ ] Commit.

## Task 3: run the grid

- [ ] `run_all.sh` with `GPU_ID` and `ARMS`, skipping runs with `config.json`. Two arms per GPU on 0, 1, 2, 3.
- [ ] Check that every arm has 5 sets × 2 noises, and the logs are free of errors.

## Task 4: collector, verdicts, figures

**Files:** create `collect_decay.py`, `test_collect_decay.py`; output `report/result.md`, `report/figures/{ce_vs_k,alpha_schedules}.png`.

**Consumes:** `collect_momentum.{ce_after_k, mean_ood_ppl, paired_bootstrap}` and the momentum results for `gd_a5`, `ema_m0.5`, `gd_a7.5`, `gd_a5`'s 2-cycle helper `analyze_q3_extra.cycle_metrics`.

- [ ] Failing tests: `test_select_d_star_excludes_ema` and `test_verdict_d1_bands` (73.2 / 78.0 bands) and `test_drift`.
- [ ] Implement: summary over k ∈ {1..10, 16, 32}, σ ∈ {0, 3}; D1–D5 table; per-set table at k = 3 with paired CIs vs `gd_a5` and vs `ema_m0.5` (assert equal `seq_id`); 2-cycle table; figure of mean OOD ppl vs k for `gd_a5`, `ema_m0.5`, `d*`, `ema_m0.5_harm`.
- [ ] Commit.

## Task 5: full-set Table 3 protocol for `d*` (only if D1 is "step size explains most" or "partial")

- [ ] Score `d*` with `../energy_momentum/table3_ema.score` on all 33,000 validation documents and every OOD item; ratio vs S2 and vs S2 + EMA with paired CIs from the existing `../energy_momentum/results/table3/*.npz`.

## Task 6: report and review

- [ ] `report/report.md` (Obsidian, 3-bullet TL;DR): setup, D1–D5 verdicts, per-set table, figures, interpretation, limitations, reproduction.
- [ ] Update `../energy_momentum/report/report.md` M2 caveat with a pointer to this result.
- [ ] Fresh-context review of the branch, fix Critical/Important, commit.
