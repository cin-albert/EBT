# MCMC Update-Rule Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> [!summary] TL;DR
> - Add `--mcmc_update_rule {probs,logits,log_probs}` (default `probs`, bit-identical to today) to the EBT NLP MCMC loop and to the thinking-study harness, through one shared helper `model/nlp/mcmc_update.py`.
> - Gate the refactor with a 150-step `probs` run against the baseline log, pick α with three 2k-step pilots (~40 GPU-min), then train S2 xxs 105k/b32 on GPU 5 one rule at a time (~10.5 h each): `logits` first, evaluated and reported, then `log_probs`.
> - Evaluate with the existing Q2/Q3 tools (made rule-aware) and a small `compare.py` that tabulates H1–H4 against the `probs` baseline.

**Goal:** Train and evaluate EBT S2 xxs under two alternative MCMC update rules (`logits`, `log_probs`) and compare them with the existing `probs` baseline on H1–H4.

**Architecture:** A pure helper computes the next logits for each rule; `EBT_NLP.forward` and `harness.mcmc` both call it, and a CPU integration test pins that the two loops agree. Training reuses the baseline's `reproduce_xxs_scaling/common.sh` so data, split, seed and batch order are identical. Analysis reuses Q2/Q3 unchanged apart from rule-aware wiring.

**Tech Stack:** Python 3.12 (`/mnt/data/conda-envs/quangnx/ebt312/bin/python`), PyTorch, PyTorch Lightning, pytest, pandas, matplotlib, bash.

**Spec:** `experiments/scale_to_16m_sequences/mcmc_update_rule/MCMC_UPDATE_RULE_ABLATION.md`

## Global Constraints

- Python only via `/mnt/data/conda-envs/quangnx/ebt312/bin/python` (written `$PY` below).
- GPU: GPU 5 only (`GPU_ID=5`). Code and tests run on CPU.
- Default rule `probs` must leave every existing run, checkpoint and analysis bit-identical.
- Data, split, seed, batch, steps: exactly the baseline's (`experiments/reproduce_xxs_scaling/common.sh`: 3.4M-doc `data/full`, `--validation_split_pct 0.0098`, seed 33, batch 32, 105k steps, 1 GPU).
- S2 flags: exactly `experiments/reproduce_xxs_scaling/run_ebt_s2_xxs.sh` (α = 5 ×[½, 2], 2–3 steps, Langevin noise 3, replay buffer, `--truncate_mcmc`, `--no_mcmc_detach`), except `--mcmc_update_rule` and α where the pilot rules otherwise.
- One training seed per rule; no extra seed runs.
- Heavy artifacts (checkpoints, CSV logs) go under `runs/` (symlink to `/mnt/data/quangnx/research/ebt/scale_16m/mcmc_update_rule/runs`) and are not committed.
- All files in English. Markdown is Obsidian-flavoured with a 3-bullet TL;DR on summary files.
- Branch `mcmc-update-rule` (from `exp/xxs-scaling`). Do not push or merge without asking. Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus

1. **Existing checkpoints have no `mcmc_update_rule` in their hparams.** Loading any of them (harness, Q2/Q3, OOD scripts) must behave as `probs`. Pinned by `test_mcmc_defaults_to_probs_when_hparams_lack_the_rule` (Task 2) and the `probs` case of `test_forward_and_harness_agree` (Task 3).
2. **α is a tensor of shape (B, S, 1)** whenever `--randomize_mcmc_step_size_scale` or `--scale_alpha_with_energy` is on (S2 training). `next_logits` must broadcast it. Pinned by `test_next_logits_broadcasts_per_token_alpha` (Task 1).
3. **Training differentiates through the update** (`create_graph=True` on the last step, `--no_mcmc_detach`). Under `logits`, model parameters must still receive finite gradients. Pinned by `test_logits_rule_backpropagates_to_the_model` (Task 3).
4. **A pilot that dies** (the forward raises on NaN gradients) leaves a short CSV. `check_runs.py pilot` must report it as failed, not crash. Pinned by `test_pilot_fails_a_run_that_stopped_early` (Task 5).
5. **Lightning CSVs mix train and validation rows** (validation rows have NaN `train_loss`). Gate and pilot checks must use train rows only. Pinned by `test_train_rows_drop_validation_rows` (Task 5).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `model/nlp/mcmc_update.py` | create | `RULES`, `next_logits`, `descent_direction` |
| `tests/test_mcmc_update.py` | create | unit tests of the helper |
| `model/nlp/ebt.py` | modify | `forward` uses `next_logits`; `_run_ebt_inference_steps` rejects non-`probs` |
| `train_model.py` | modify | `--mcmc_update_rule` flag and its validation |
| `experiments/thinking_study/common/harness.py` | modify | `mcmc` uses the model's rule; new `RUN_DIRS` keys |
| `experiments/thinking_study/common/fake_ebt.py` | create | tiny CPU stand-in model for harness/Q3 tests (test utility) |
| `experiments/thinking_study/common/test_harness.py` | modify | rule tests for `mcmc` |
| `experiments/thinking_study/common/test_rule_consistency.py` | create | forward ↔ harness agreement on the S2 105k checkpoint (CPU) |
| `experiments/thinking_study/q3_energy_landscape/run_q3.py` | modify | rule-aware `gnorm`, line profile, Jacobian skip |
| `experiments/thinking_study/q3_energy_landscape/test_run_q3.py` | create | line-profile endpoint test |
| `.../mcmc_update_rule/common.sh`, `run_rule.sh`, `pilot.sh`, `full.sh` | create | launch scripts |
| `.../mcmc_update_rule/check_runs.py` + `test_check_runs.py` | create | gate and pilot verdicts from CSVs |
| `.../mcmc_update_rule/eval.sh` | create | Q2/Q3 for one or more rule runs |
| `.../mcmc_update_rule/compare.py` + `test_compare.py` | create | H1–H4 table and CE-vs-k figure over whichever runs exist |
| `.../mcmc_update_rule/report/report.md` | create | `logits` results first (Task 8), extended with `log_probs` (Task 10) |

`...` = `experiments/scale_to_16m_sequences`. All commands below run from the repo root `refs/EBT` unless stated.

---

### Task 1: Update-rule helper

**Files:**
- Create: `model/nlp/mcmc_update.py`
- Test: `tests/test_mcmc_update.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RULES: tuple[str, ...] = ("probs", "logits", "log_probs")`
  - `next_logits(rule: str, z: Tensor, p: Tensor, grad_p: Tensor, alpha: float | Tensor) -> Tensor` — `z` is the (noisy) logits the step started from, `p = softmax(z)`, `grad_p = dE/dp`; all (..., V); `alpha` scalar or broadcastable (..., 1).
  - `descent_direction(rule: str, p: Tensor, grad_p: Tensor) -> Tensor` — the direction the rule moves along with the softmax-null component removed; its norm is the stationarity measure (H4).
  - Both raise `ValueError` on an unknown rule.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch

from model.nlp.mcmc_update import RULES, descent_direction, next_logits

V = 11


def _point(seed=0, shape=(2, 3)):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(*shape, V, generator=g, dtype=torch.float64)
    w = torch.randn(V, 4, generator=g, dtype=torch.float64)
    return z, w


def _energy(p, w):
    return ((p @ w) ** 2).sum(-1)  # any smooth E(p) works


def test_probs_rule_is_the_repo_update():
    z, w = _point()
    p = torch.softmax(z, -1).requires_grad_()
    (g,) = torch.autograd.grad(_energy(p, w).sum(), p)
    assert torch.equal(next_logits("probs", z, p, g, 5.0), p - 5.0 * g)


def test_logits_rule_matches_autograd_through_softmax():
    z, w = _point()
    z = z.requires_grad_()
    p = torch.softmax(z, -1)
    g_p, g_z = torch.autograd.grad(_energy(p, w).sum(), [p, z])
    out = next_logits("logits", z.detach(), p.detach(), g_p, 0.7)
    assert torch.allclose(out, z.detach() - 0.7 * g_z, atol=1e-12)


def test_log_probs_rule_keeps_p_where_the_gradient_is_constant():
    z, _ = _point()
    p = torch.softmax(z, -1)
    g = torch.full_like(p, 2.5)
    assert torch.allclose(torch.softmax(next_logits("log_probs", z, p, g, 5.0), -1), p, atol=1e-12)


def test_probs_rule_moves_off_a_non_uniform_point_even_with_constant_gradient():
    z, _ = _point()
    p = torch.softmax(z, -1)
    g = torch.full_like(p, 2.5)
    assert not torch.allclose(torch.softmax(next_logits("probs", z, p, g, 5.0), -1), p, atol=1e-3)


@pytest.mark.parametrize("rule", RULES)
def test_descent_direction_vanishes_for_a_constant_gradient(rule):
    z, _ = _point()
    p = torch.softmax(z, -1)
    assert torch.allclose(descent_direction(rule, p, torch.full_like(p, -1.3)), torch.zeros_like(p), atol=1e-12)


def test_descent_direction_of_logits_is_the_logit_gradient():
    z, w = _point()
    z = z.requires_grad_()
    p = torch.softmax(z, -1)
    g_p, g_z = torch.autograd.grad(_energy(p, w).sum(), [p, z])
    assert torch.allclose(descent_direction("logits", p.detach(), g_p), g_z, atol=1e-12)


@pytest.mark.parametrize("rule", RULES)
def test_next_logits_broadcasts_per_token_alpha(rule):
    z, w = _point()
    p = torch.softmax(z, -1).requires_grad_()
    (g,) = torch.autograd.grad(_energy(p, w).sum(), p)
    alpha = torch.rand(2, 3, 1, dtype=torch.float64) + 0.5
    out = next_logits(rule, z, p.detach(), g, alpha)
    one = next_logits(rule, z[0, 1], p.detach()[0, 1], g[0, 1], float(alpha[0, 1]))
    assert out.shape == z.shape and torch.allclose(out[0, 1], one, atol=1e-12)


def test_unknown_rule_raises():
    z, _ = _point()
    p = torch.softmax(z, -1)
    with pytest.raises(ValueError):
        next_logits("energy", z, p, p, 1.0)
    with pytest.raises(ValueError):
        descent_direction("energy", p, p)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_mcmc_update.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'model.nlp.mcmc_update'`.

- [ ] **Step 3: Write the implementation**

```python
"""Update rules of the EBT NLP MCMC loop: next logits from the step's (noisy) logits z, p = softmax(z) and g = dE/dp.

- probs (repo default): z' = p - alpha * g. Fixed points are the stationary points on the simplex of
  alpha * E(p) - H(p) - 0.5 * ||p||^2 (docs/EXPLAIN_FIXED_POINT.md), not of E.
- logits (paper pseudocode): z' = z - alpha * dE/dz, with dE/dz = p * (g - <p, g>) by the chain rule through softmax.
- log_probs (exponentiated gradient, mirror descent on the simplex): z' = log p - alpha * g. Fixed points are the
  stationary points of E on the simplex.
"""
import torch

RULES = ("probs", "logits", "log_probs")


def _check(rule: str) -> None:
    if rule not in RULES:
        raise ValueError(f"unknown MCMC update rule {rule!r}; expected one of {RULES}")


def descent_direction(rule: str, p: torch.Tensor, grad_p: torch.Tensor) -> torch.Tensor:
    """Direction the rule moves along, without the component softmax ignores; zero exactly at its fixed points."""
    _check(rule)
    if rule == "logits":
        return p * (grad_p - (p * grad_p).sum(-1, keepdim=True))
    return grad_p - grad_p.mean(-1, keepdim=True)


def next_logits(rule: str, z: torch.Tensor, p: torch.Tensor, grad_p: torch.Tensor, alpha) -> torch.Tensor:
    _check(rule)
    if rule == "probs":
        return p - alpha * grad_p
    if rule == "log_probs":
        return torch.log_softmax(z, dim=-1) - alpha * grad_p
    return z - alpha * descent_direction(rule, p, grad_p)
```

Note: `descent_direction("probs", ...)` is not the vector `next_logits("probs", ...)` subtracts; it differs by a per-token constant that softmax ignores. `next_logits("probs")` keeps the exact repo expression so the default path stays bit-identical.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/test_mcmc_update.py -q`
Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add model/nlp/mcmc_update.py tests/test_mcmc_update.py experiments/scale_to_16m_sequences/mcmc_update_rule/MCMC_UPDATE_RULE_ABLATION.md experiments/scale_to_16m_sequences/mcmc_update_rule/IMPLEMENTATION_PLAN.md
git commit -m "feat: MCMC update-rule helper (probs, logits, log_probs) with spec and plan

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Rule-aware harness loop

**Files:**
- Create: `experiments/thinking_study/common/fake_ebt.py`
- Modify: `experiments/thinking_study/common/harness.py` (docstring `:1-11`, `RUN_DIRS` `:22-30`, `mcmc` `:137-158`)
- Test: `experiments/thinking_study/common/test_harness.py`

**Interfaces:**
- Consumes: `next_logits` from Task 1.
- Produces:
  - `harness.mcmc(...)` / `harness.run_mcmc(...)`: same signatures; the update now follows `model.hparams.get("mcmc_update_rule", "probs")`.
  - `harness.RUN_DIRS` keys `"s2_rule_logits"` → run name `s2-rule-logits-105k`, `"s2_rule_log_probs"` → `s2-rule-log-probs-105k`, root `RULE_CKPTS = REPO / "experiments/scale_to_16m_sequences/mcmc_update_rule/runs/logs/checkpoints"`.
  - `fake_ebt.FakeEBT(rule: str | None, n_steps: int = 2)`: a CPU module with `embeddings`, `vocab_to_embed`, `transformer(x, start_pos, mcmc_step)`, `corrupt_embeddings`, `alpha`, `hparams` (an `AttributeDict`-like dict), `V = 9`, `D = 4`. Test utility only.

- [ ] **Step 1: Write the test utility and failing tests**

`experiments/thinking_study/common/fake_ebt.py`:

```python
"""Tiny CPU stand-in for EBT_NLP with the attributes harness.mcmc and Q3 use (test utility)."""
import torch
from lightning_fabric.utilities.data import AttributeDict
from torch import nn

V, D = 9, 4


class _Energy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(D, D, generator=torch.Generator().manual_seed(1)))

    def forward(self, x, start_pos=0, mcmc_step=0):  # (B, 2S, D) -> per-token energy of the predicted half, (B, S, 1)
        pred = x[:, x.shape[1] // 2:]
        return ((pred @ self.w) ** 2).sum(-1, keepdim=True) * (1 + mcmc_step)


class FakeEBT(nn.Module):
    def __init__(self, rule=None, n_steps=2):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.embeddings = nn.Embedding(V, D)
        self.vocab_to_embed = nn.Linear(V, D, bias=False)
        with torch.no_grad():
            self.embeddings.weight.copy_(torch.randn(V, D, generator=g))
            self.vocab_to_embed.weight.copy_(torch.randn(D, V, generator=g))
        self.transformer = _Energy()
        self.alpha = nn.Parameter(torch.tensor(0.8), requires_grad=False)
        hp = {"mcmc_num_steps": n_steps}
        if rule is not None:
            hp["mcmc_update_rule"] = rule
        self.hparams = AttributeDict(hp)

    def corrupt_embeddings(self, emb):
        return torch.randn(*emb.shape[:2], V)
```

If `lightning_fabric.utilities.data.AttributeDict` does not import in this env, use `from pytorch_lightning.utilities.parsing import AttributeDict` (the class `self.hparams` is in Lightning modules).

Append to `experiments/thinking_study/common/test_harness.py`:

```python
import pytest
import torch
import torch.nn.functional as F

from fake_ebt import FakeEBT
from model.nlp.mcmc_update import RULES, next_logits


def _final_z(model, n_steps=2):
    torch.manual_seed(0)
    return harness.run_mcmc(model, torch.tensor([[1, 2, 3]]), n_steps, noise_std=0.0)


def _manual_final_z(model, n_steps, rule):
    torch.manual_seed(0)
    ctx = model.embeddings(torch.tensor([[1, 2, 3]]))
    z = model.corrupt_embeddings(ctx)
    for t in range(n_steps):
        p = F.softmax(z, -1).requires_grad_()
        e = harness.energy(model, ctx, p, harness.landscape_index(model, t))
        (g,) = torch.autograd.grad(e.sum(), p)
        z = next_logits(rule, z, p.detach(), g, float(model.alpha))
    return z.detach()


@pytest.mark.parametrize("rule", RULES)
def test_mcmc_applies_the_models_rule(rule):
    model = FakeEBT(rule)
    assert torch.allclose(_final_z(model), _manual_final_z(model, 2, rule), atol=1e-6)


def test_mcmc_defaults_to_probs_when_hparams_lack_the_rule():
    assert torch.allclose(_final_z(FakeEBT(None)), _manual_final_z(FakeEBT(None), 2, "probs"), atol=1e-6)


def test_rule_runs_resolve_to_the_rule_checkpoint_root(tmp_path, monkeypatch):
    run = tmp_path / "s2-rule-log-probs-105k_2026-11-01_00-00-00_"
    run.mkdir()
    (run / "epoch=epoch=0-step=step=105000-valid_loss=valid_loss=3.9.ckpt").touch()
    monkeypatch.setitem(harness.RUN_DIRS, "s2_rule_log_probs", (tmp_path, harness.RUN_DIRS["s2_rule_log_probs"][1]))
    assert harness.ckpt_path("s2_rule_log_probs").parent == run
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd experiments/thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -m pytest common/test_harness.py -q`
Expected: `test_mcmc_applies_the_models_rule[logits]` and `[log_probs]` FAIL (harness always applies `probs`); `test_rule_runs_resolve_to_the_rule_checkpoint_root` FAILS with `KeyError: 's2_rule_log_probs'`; `[probs]` and the default test pass.

- [ ] **Step 3: Implement**

In `harness.py`:
- Docstring bullet `:7-8` becomes: `- each step: p_t = softmax(z_t + Langevin noise); E_t = E(p_t); g_t = dE/dp_t; z_{t+1} = next_logits(rule, z_t + noise, p_t, g_t, alpha) (model/nlp/mcmc_update.py); rule = hparams mcmc_update_rule, default probs: z_{t+1} = p_t - alpha * g_t`.
- Add the import `from model.nlp.mcmc_update import next_logits` after the torch imports.
- After `SCALE_CKPTS`, add `RULE_CKPTS = REPO / "experiments/scale_to_16m_sequences/mcmc_update_rule/runs/logs/checkpoints"` and to `RUN_DIRS`:

```python
    "s2_rule_logits": (RULE_CKPTS, "s2-rule-logits-105k_20*"),
    "s2_rule_log_probs": (RULE_CKPTS, "s2-rule-log-probs-105k_20*"),
```

- In `mcmc`, read the rule once and replace the update line:

```python
    rule = model.hparams.get("mcmc_update_rule", "probs")  # checkpoints from before the flag are probs
    ...
        yield Step(t, z, p.detach(), e.detach(), g, lsc)
        z = next_logits(rule, noisy, p.detach(), g, alpha)
```

For `probs` this is `p.detach() - alpha * g`, the old line.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd experiments/thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -m pytest common -q`
Expected: all pass (the 4 old harness tests, the 5 new ones, and the OOD tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/thinking_study/common/harness.py experiments/thinking_study/common/fake_ebt.py experiments/thinking_study/common/test_harness.py
git commit -m "feat: harness MCMC loop follows the checkpoint's update rule; rule run keys

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Rule-aware training loop and flag

**Files:**
- Modify: `model/nlp/ebt.py` (`forward` `:115-171`, `_run_ebt_inference_steps` `:442`)
- Modify: `train_model.py` (NLP validation block `:99-102`; argument next to `--mcmc_step_size` `:366`)
- Test: `experiments/thinking_study/common/test_rule_consistency.py`

**Interfaces:**
- Consumes: `next_logits` (Task 1); rule-aware `harness.run_mcmc`, `harness.load_model` (Task 2).
- Produces: `--mcmc_update_rule {probs,logits,log_probs}` (default `probs`), stored in checkpoint hparams; `EBT_NLP.forward` reads `self.hparams.get("mcmc_update_rule", "probs")` on every call.

- [ ] **Step 1: Write the failing tests**

`experiments/thinking_study/common/test_rule_consistency.py`:

```python
"""EBT_NLP.forward and harness.mcmc must apply the same update rule (S2 105k checkpoint on CPU)."""
import pytest
import torch

import harness
from model.nlp.mcmc_update import RULES


@pytest.fixture(scope="module")
def s2():
    try:
        harness.ckpt_path("s2")
    except FileNotFoundError:
        pytest.skip("S2 105k checkpoint not available")
    model, _ = harness.load_model("s2", device="cpu")
    return model


@pytest.mark.parametrize("rule", RULES)
def test_forward_and_harness_agree(s2, rule):
    s2.hparams.mcmc_update_rule = rule
    x = torch.randint(1, 1000, (2, 16), generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    dists, _ = s2(x, learning=False, no_randomness=True)
    torch.manual_seed(0)
    z = harness.run_mcmc(s2, x, len(dists), float(s2.langevin_dynamics_noise_std))
    assert torch.allclose(dists[-1].reshape(2, 16, -1), torch.log_softmax(z, -1), atol=1e-4)


def test_logits_rule_backpropagates_to_the_model(s2):
    s2.hparams.mcmc_update_rule = "logits"
    s2.zero_grad()
    x = torch.randint(1, 1000, (1, 8), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    dists, _ = s2(x, learning=True, no_randomness=True)
    dists[-1].mean().backward()
    grads = [p.grad for p in s2.transformer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)


def test_advanced_inference_rejects_other_rules(s2):
    s2.hparams.mcmc_update_rule = "log_probs"
    with pytest.raises(NotImplementedError):
        s2._run_ebt_inference_steps(None, None, None, None, 0, False)
```

The fixture is module-scoped and each test sets the rule it needs, so test order does not matter.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd experiments/thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -m pytest common/test_rule_consistency.py -q`
Expected: `[probs]` passes; `[logits]`, `[log_probs]` FAIL on `allclose` (forward still applies `probs`); `test_advanced_inference_rejects_other_rules` FAILS (no error raised; it fails on `None` inputs with a different exception, or not at all). `test_logits_rule_backpropagates_to_the_model` may already pass, since forward ignores the rule; it guards the change in Step 3.

- [ ] **Step 3: Implement**

`model/nlp/ebt.py`:
- Import at the top with the other local imports: `from model.nlp.mcmc_update import next_logits`.
- In `forward`, read the rule before the loop: `rule = self.hparams.get("mcmc_update_rule", "probs")  # checkpoints from before the flag are probs`.
- Right after the Langevin noise block (`:123-125`), keep the noisy logits: `noisy_logits = predicted_tokens  # logits this step starts from; the logits and log_probs rules update from them`.
- Replace the `scale_alpha_with_energy` if/else update (`:164-171`) with:

```python
                if self.hparams.scale_alpha_with_energy:
                    # with truncate_mcmc, non-final steps use create_graph=False which frees energy_preds's backward state; we detach here so the step-size modulation doesn't dangle into freed graph
                    energy_for_scaling = energy_preds.detach() if (self.hparams.truncate_mcmc and i != (len(mcmc_steps) - 1)) else energy_preds
                    exponentiated_energies = torch.exp(energy_for_scaling.reshape(batch_size, seq_length, 1) / self.hparams.scale_alpha_with_energy_temp)
                    step_alpha = alpha * exponentiated_energies
                else:
                    step_alpha = alpha
                predicted_tokens = next_logits(rule, noisy_logits, predicted_tokens, predicted_tokens_grad, step_alpha) # probs (default): p - alpha * dE/dp, an unnormalized prob dist converted to a prob dist after
```

- First line of `_run_ebt_inference_steps` body:

```python
        if self.hparams.get("mcmc_update_rule", "probs") != "probs":
            raise NotImplementedError("advanced inference implements only the probs MCMC update rule")
```

`train_model.py`:
- After `--mcmc_step_size_learnable` (`:368`):

```python
    parser.add_argument("--mcmc_update_rule", help="how an MCMC step forms the next logits from p = softmax(z) and dE/dp: probs (repo default, z' = p - a*dE/dp), logits (paper pseudocode, z' = z - a*dE/dz), log_probs (exponentiated gradient, z' = log p - a*dE/dp); see model/nlp/mcmc_update.py", type=str, default="probs", choices=["probs", "logits", "log_probs"])
```

- In the NLP branch of the validation block (`:99-102`):

```python
        if args.mcmc_update_rule != "probs":
            assert args.normalize_initial_condition and not args.normalize_initial_condition_only_first_step, "mcmc_update_rule logits/log_probs assume p = softmax(z) at every MCMC step"
```

- [ ] **Step 4: Run all tests**

Run: `cd experiments/thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -m pytest common -q && cd ../.. && $PY -m pytest tests -q`
Expected: all pass, including the 5 tests of `test_rule_consistency.py`; the `tests/` suite is unchanged (plus 12 from Task 1).

- [ ] **Step 5: Commit**

```bash
git add model/nlp/ebt.py train_model.py experiments/thinking_study/common/test_rule_consistency.py
git commit -m "feat: --mcmc_update_rule in the EBT NLP training loop (default probs unchanged)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Rule-aware Q3

**Files:**
- Modify: `experiments/thinking_study/q3_energy_landscape/run_q3.py` (docstring, `part_a` `:40-63`, `hessian_and_jacobian` `:83-125`, `line_profile` `:130-141`, `part_b`, `main`)
- Test: `experiments/thinking_study/q3_energy_landscape/test_run_q3.py`

**Interfaces:**
- Consumes: `next_logits`, `descent_direction` (Task 1); rule-aware `harness.mcmc` (Task 2); `fake_ebt.FakeEBT` (Task 2).
- Produces: `line_profile(model, ctx, s, alpha, targets, mask, rule="probs")`, `hessian_and_jacobian(model, ctx, s, alpha, mask, rule="probs", n_power=30)`, `part_a(..., rule="probs")`, `part_b(..., rule="probs")`; `config.json` gains `"mcmc_update_rule"`. Output arrays keep their names; `gnorm` is now `‖descent_direction(rule, p, g)‖` (unchanged for `probs`); `t{t}_rho` is NaN for other rules.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pytest
import torch

import run_q3
from fake_ebt import FakeEBT
from harness import mcmc, token_logprob
from model.nlp.mcmc_update import RULES, next_logits


@pytest.mark.parametrize("rule", RULES)
def test_line_profile_at_s1_is_the_rules_actual_update(rule):
    model = FakeEBT(rule)
    x = torch.tensor([[1, 2, 3]])
    targets, mask = torch.tensor([[4, 5, 6]]), torch.ones(1, 3, dtype=torch.bool)
    torch.manual_seed(0)
    s = next(mcmc(model, x, 2, 0.0))
    ctx = model.embeddings(x).detach()
    line = run_q3.line_profile(model, ctx, s, float(model.alpha), targets, mask, rule)
    i = int(np.argmin(np.abs(run_q3.LINE_S - 1.0)))
    expected = token_logprob(next_logits(rule, s.z, s.p, s.grad, float(model.alpha)), targets)[mask]
    assert torch.allclose(line["logp"][i], expected, atol=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd experiments/thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -m pytest q3_energy_landscape/test_run_q3.py -q`
Expected: FAIL with `TypeError: line_profile() takes 6 positional arguments but 7 were given`.

- [ ] **Step 3: Implement**

- Docstring: after the Phi paragraph add `Rules other than probs (hparams mcmc_update_rule, model/nlp/mcmc_update.py): gnorm and the line profile follow the rule; Phi is recorded but only meaningful for probs; the Jacobian spectral radius is computed only for probs (NaN otherwise).`
- Import: `from model.nlp.mcmc_update import descent_direction, next_logits`.
- `part_a(model, val_ds, idx, n_steps, noise, alpha, batch_size, seed, rule="probs")`: replace `g = s.grad - s.grad.mean(-1, keepdim=True)  # the component softmax ignores is removed` with `g = descent_direction(rule, s.p, s.grad)  # the rule's step direction, softmax-null component removed`.
- `hessian_and_jacobian(model, ctx, s, alpha, mask, rule="probs", n_power=30)`: wrap the power iteration (from `q = F.softmax(...)` to the end of the `for` loop) in `if rule == "probs":`, with `else: log_growth = torch.full(e.shape[:2], float("nan"), device=e.device)  # T of the other rules is not derived here` and `n_power` division unchanged (NaN stays NaN).
- `line_profile(model, ctx, s, alpha, targets, mask, rule="probs")`: replace `z = s.p - sc * alpha * s.grad` with `z = next_logits(rule, s.z, s.p, s.grad, sc * alpha)  # Part B runs at noise 0, so s.z is the logits the step starts from`.
- `part_b(..., seed, rule="probs")`: pass `rule` to both calls.
- `main`: `rule = model.hparams.get("mcmc_update_rule", "probs")`; pass `rule=rule` to `part_a` and `part_b`; add `"mcmc_update_rule": rule` to `config`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd experiments/thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -m pytest q3_energy_landscape common -q`
Expected: all pass (3 new).

- [ ] **Step 5: Commit**

```bash
git add experiments/thinking_study/q3_energy_landscape/run_q3.py experiments/thinking_study/q3_energy_landscape/test_run_q3.py
git commit -m "feat: Q3 follows the checkpoint's MCMC update rule

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Launch scripts and run checks

**Files (all under `experiments/scale_to_16m_sequences/mcmc_update_rule/`):**
- Create: `common.sh`, `run_rule.sh`, `pilot.sh`, `full.sh`, `check_runs.py`
- Test: `test_check_runs.py`

**Interfaces:**
- Consumes: `--mcmc_update_rule` (Task 3).
- Produces:
  - `GPU_ID=5 ./run_rule.sh RULE ALPHA RUN_NAME [extra train_model.py args]`; CSV at `runs/logs/csv/RUN_NAME/version_0/metrics.csv`.
  - `check_runs.py gate CSV` (exit 0 iff gate passes), `check_runs.py pilot CSV...` (prints a JSON verdict per run).
  - Functions: `train_rows(csv) -> DataFrame` (indexed by step), `gate(run, base) -> dict`, `mcmc_gain(rows) -> float`, `pilot_verdict(run, base) -> dict` with keys `finite, completed, gain, baseline_gain, final_step_loss, passed`.

- [ ] **Step 1: Write the failing tests**

`test_check_runs.py`:

```python
import numpy as np
import pandas as pd

import check_runs


def _csv(tmp_path, name, steps, initial, final, extra_val_rows=True):
    rows = [{"step": s, "train_loss": f, "train_initial_loss": i, "train_final_step_loss": f, "valid_loss": np.nan}
            for s, i, f in zip(steps, initial, final)]
    if extra_val_rows:
        rows.append({"step": steps[-1], "train_loss": np.nan, "train_initial_loss": np.nan,
                     "train_final_step_loss": np.nan, "valid_loss": 4.0})
    path = tmp_path / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


STEPS = list(range(49, 2000, 50))


def test_train_rows_drop_validation_rows(tmp_path):
    rows = check_runs.train_rows(_csv(tmp_path, "a", STEPS, [8.0] * 40, [6.0] * 40))
    assert len(rows) == 40 and rows["train_loss"].notna().all()


def test_gate_passes_within_tolerance_and_fails_outside(tmp_path):
    base = _csv(tmp_path, "base", [49, 99, 149], [9, 9, 9], [30.0, 13.0, 10.0])
    ok = _csv(tmp_path, "ok", [49, 99, 149], [9, 9, 9], [30.3, 13.1, 10.05])
    bad = _csv(tmp_path, "bad", [49, 99, 149], [9, 9, 9], [30.0, 14.0, 10.0])
    assert check_runs.gate(ok, base)["passed"] and not check_runs.gate(bad, base)["passed"]


def test_mcmc_gain_averages_the_window(tmp_path):
    rows = check_runs.train_rows(_csv(tmp_path, "a", STEPS, [8.0] * 30 + [9.0] * 10, [6.0] * 40))
    assert check_runs.mcmc_gain(rows) == 3.0  # steps 1549..1999 are the last 10 rows


def test_pilot_passes_with_half_the_baseline_gain(tmp_path):
    base = _csv(tmp_path, "base", STEPS, [8.0] * 40, [6.0] * 40)
    run = _csv(tmp_path, "run", STEPS, [8.0] * 40, [7.0] * 40)
    weak = _csv(tmp_path, "weak", STEPS, [8.0] * 40, [7.5] * 40)
    assert check_runs.pilot_verdict(run, base)["passed"]
    assert not check_runs.pilot_verdict(weak, base)["passed"]


def test_pilot_fails_a_run_that_stopped_early(tmp_path):
    base = _csv(tmp_path, "base", STEPS, [8.0] * 40, [6.0] * 40)
    short = _csv(tmp_path, "short", STEPS[:20], [8.0] * 20, [5.0] * 20)
    v = check_runs.pilot_verdict(short, base)
    assert not v["completed"] and not v["passed"]


def test_pilot_fails_a_run_with_nan_loss(tmp_path):
    base = _csv(tmp_path, "base", STEPS, [8.0] * 40, [6.0] * 40)
    nan = _csv(tmp_path, "nan", STEPS, [8.0] * 40, [6.0] * 39 + [np.nan], extra_val_rows=False)
    assert not check_runs.pilot_verdict(nan, base)["passed"]
```

In the NaN test the last row's `train_loss` is NaN, so `train_rows` drops it and the run looks short; `finite` must therefore be checked on the raw CSV rows that have a `train_initial_loss`, before dropping (see the implementation).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd experiments/scale_to_16m_sequences/mcmc_update_rule && $PY -m pytest test_check_runs.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'check_runs'`.

- [ ] **Step 3: Implement**

`check_runs.py`:

```python
"""Verdicts from Lightning metrics.csv files: the probs gate against the baseline, and the α pilots.

gate:  the refactored probs path reproduces the baseline's train loss at its first logged steps.
pilot: the MCMC loop does work under a new rule: mean (CE after the first step - CE after the last step)
       over logged steps 1,500-2,000 is at least half the baseline's, with no NaN/Inf and the run complete.

Usage: python check_runs.py gate RUN_CSV | python check_runs.py pilot RUN_CSV [RUN_CSV ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BASELINE_CSV = REPO / "experiments/reproduce_xxs_scaling/runs/logs/csv/ebt-s2-xxs-scaling-105k/version_0/metrics.csv"
GATE_STEPS, GATE_RTOL = (49, 99, 149), 0.02
WINDOW, LAST_STEP, MIN_GAIN_RATIO = (1500, 2000), 1999, 0.5


def train_rows(csv) -> pd.DataFrame:
    d = pd.read_csv(csv)
    return d[d["train_loss"].notna()].set_index("step")


def gate(run, base=BASELINE_CSV) -> dict:
    r, b = train_rows(run), train_rows(base)
    steps = {s: (float(r.at[s, "train_loss"]), float(b.at[s, "train_loss"])) for s in GATE_STEPS if s in r.index}
    ok = len(steps) == len(GATE_STEPS) and all(abs(x - y) <= GATE_RTOL * abs(y) for x, y in steps.values())
    return {"steps": steps, "passed": ok}


def mcmc_gain(rows: pd.DataFrame) -> float:
    w = rows[(rows.index >= WINDOW[0]) & (rows.index <= WINDOW[1])]
    return float((w["train_initial_loss"] - w["train_final_step_loss"]).mean())


def pilot_verdict(run, base=BASELINE_CSV) -> dict:
    raw = pd.read_csv(run)
    logged = raw[raw["train_initial_loss"].notna()]
    finite = bool(np.isfinite(logged[["train_loss", "train_final_step_loss"]].to_numpy()).all())
    rows, b = train_rows(run), train_rows(base)
    completed = bool(len(rows)) and int(rows.index.max()) >= LAST_STEP
    gain, base_gain = mcmc_gain(rows), mcmc_gain(b)
    w = rows[(rows.index >= WINDOW[0]) & (rows.index <= WINDOW[1])]
    return {"finite": finite, "completed": completed, "gain": gain, "baseline_gain": base_gain,
            "final_step_loss": float(w["train_final_step_loss"].mean()),
            "passed": finite and completed and gain >= MIN_GAIN_RATIO * base_gain}


def main():
    cmd, paths = sys.argv[1], sys.argv[2:]
    if cmd == "gate":
        v = gate(paths[0])
        print(json.dumps(v, indent=1))
        sys.exit(0 if v["passed"] else 1)
    print(json.dumps({p: pilot_verdict(p) for p in paths}, indent=1))


if __name__ == "__main__":
    main()
```

`common.sh`:

```bash
#!/usr/bin/env bash
# Baseline settings (../../reproduce_xxs_scaling/common.sh: xxs, 1 GPU, batch 32, 105k steps, 3.4M-doc data, seed 33)
# with this workspace's runs/ and logs/, plus the S2 flags of ../../reproduce_xxs_scaling/run_ebt_s2_xxs.sh.
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WS/../../reproduce_xxs_scaling/common.sh"  # sets PY, REPO, COMMON_ARGS, CUDA_VISIBLE_DEVICES; checks GPU_ID is idle
mkdir -p "$WS/runs" "$WS/logs"
export TRACKIO_DIR="$WS/runs/trackio"
cd "$WS/runs"  # train_model.py writes ./logs/{checkpoints,csv}

S2_ARGS=(
  --model_name ebt
  --normalize_initial_condition --ebt_type time_embed --denoising_initial_condition random_noise
  --no_mcmc_detach --mcmc_replay_buffer --truncate_mcmc --langevin_dynamics_noise 3.0
  --randomize_mcmc_step_size_scale 2
  --mcmc_num_steps 1 --randomize_mcmc_num_steps 2 --randomize_mcmc_num_steps_min 2
  --wandb_project ebt_mcmc_update_rule
)
```

`run_rule.sh`:

```bash
#!/usr/bin/env bash
# S2 xxs with a given MCMC update rule and step size; everything else as the probs baseline.
# Usage: GPU_ID=5 ./run_rule.sh RULE ALPHA RUN_NAME [extra train_model.py args]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
rule="$1" alpha="$2" name="$3"
shift 3
exec "$PY" "$REPO/train_model.py" --run_name "$name" "${COMMON_ARGS[@]}" "${S2_ARGS[@]}" \
  --mcmc_update_rule "$rule" --mcmc_step_size "$alpha" "$@"
```

`pilot.sh`:

```bash
#!/usr/bin/env bash
# Gate (probs, 150 steps, must match the baseline log), then the three 2k-step α pilots; one GPU, sequential.
# Usage: GPU_ID=5 ./pilot.sh
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
FAST=(--limit_val_batches 0 --save_top_k_ckpts 0)
csv() { echo "$WS/runs/logs/csv/$1/version_0/metrics.csv"; }

"$WS/run_rule.sh" probs 5 gate-probs-150 --max_steps 150 "${FAST[@]}" > "$WS/logs/gate-probs-150.log" 2>&1
"$PY" "$WS/check_runs.py" gate "$(csv gate-probs-150)" | tee "$WS/logs/gate.json"

names=()
for cfg in "logits 5" "logits 500" "log_probs 5"; do
  read -r rule alpha <<< "$cfg"
  name="pilot-${rule//_/-}-a${alpha}"
  names+=("$(csv "$name")")
  "$WS/run_rule.sh" "$rule" "$alpha" "$name" --max_steps 2000 "${FAST[@]}" > "$WS/logs/$name.log" 2>&1 \
    || echo "$name exited with status $?"
done
"$PY" "$WS/check_runs.py" pilot "${names[@]}" | tee "$WS/logs/pilot.json"
```

`full.sh`:

```bash
#!/usr/bin/env bash
# Full 105k runs, one per "RULE ALPHA" argument, sequential on one GPU; checkpoints every 15k (all kept).
# Usage: GPU_ID=5 ./full.sh "logits 500" "log_probs 5"
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for cfg in "$@"; do
  read -r rule alpha <<< "$cfg"
  name="s2-rule-${rule//_/-}-105k"
  "$WS/run_rule.sh" "$rule" "$alpha" "$name" --save_top_k_ckpts -1 > "$WS/logs/$name.log" 2>&1
done
```

`chmod +x common.sh run_rule.sh pilot.sh full.sh`.

- [ ] **Step 4: Run the tests and a dry check**

Run: `cd experiments/scale_to_16m_sequences/mcmc_update_rule && $PY -m pytest test_check_runs.py -q && $PY check_runs.py pilot ../../reproduce_xxs_scaling/runs/logs/csv/ebt-s2-xxs-scaling-105k/version_0/metrics.csv`
Expected: `6 passed`; the baseline against itself gives `"gain": 1.79...`, `"passed": true`.

Run: `bash -n common.sh run_rule.sh pilot.sh full.sh`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add experiments/scale_to_16m_sequences/mcmc_update_rule/{common.sh,run_rule.sh,pilot.sh,full.sh,check_runs.py,test_check_runs.py}
git commit -m "feat: update-rule ablation launch scripts, gate and pilot checks

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Gate and pilot (GPU 5, ~45 min)

**Files:**
- Create: `experiments/scale_to_16m_sequences/mcmc_update_rule/results/pilot.json` (copy of `logs/pilot.json`, committed), `results/gate.json`

**Interfaces:**
- Consumes: `pilot.sh`, `check_runs.py` (Task 5).
- Produces: the chosen α per rule for Tasks 7 and 9, recorded in the ledger as a `Ruling:` and in `results/pilot.json`.

- [ ] **Step 1: Check GPU 5 is idle and launch in the background**

Run: `nvidia-smi --query-gpu=index,memory.used --format=csv -i 5`
Expected: memory used < 1000 MiB. If not, stop and ask the user.

Run (background): `cd experiments/scale_to_16m_sequences/mcmc_update_rule && GPU_ID=5 ./pilot.sh > logs/pilot_driver.log 2>&1`
Expected: finishes in ~45 min. Wait for the completion notification; do not poll.

- [ ] **Step 2: Read the gate**

Run: `cat logs/gate.json`
Expected: `"passed": true`; the three train losses within 2% of 31.43 / 13.43 / 10.17. If the gate fails, `pilot.sh` exits before the pilots: use superpowers:systematic-debugging (first suspect: a flag differs from `run_ebt_s2_xxs.sh`, or the `probs` path is no longer bit-identical) and do not proceed.

- [ ] **Step 3: Read the pilot verdicts and rule on α**

Run: `cat logs/pilot.json`
Expected: one verdict per run with `gain`, `baseline_gain` (≈ 1.79) and `passed`.

Apply the spec's selection (§4.2):
- `logits`: both passed → lower `final_step_loss`; one passed → that one; none → no full `logits` run (record as a result).
- `log_probs`: α = 5 passed → use it; failed → run `GPU_ID=5 ./run_rule.sh log_probs 1 pilot-log-probs-a1 --max_steps 2000 --limit_val_batches 0 --save_top_k_ckpts 0` and re-check with `check_runs.py pilot`; failed again → no full `log_probs` run.
- Ledger: `Task 6: Ruling: full runs = <list of "rule α"> — <verdict numbers> — cost if wrong: one 10.5 h run on the wrong α`.

- [ ] **Step 4: Commit the verdicts**

```bash
mkdir -p experiments/scale_to_16m_sequences/mcmc_update_rule/results
cp experiments/scale_to_16m_sequences/mcmc_update_rule/logs/{gate,pilot}.json experiments/scale_to_16m_sequences/mcmc_update_rule/results/
git add -f experiments/scale_to_16m_sequences/mcmc_update_rule/results/{gate,pilot}.json
git commit -m "results: update-rule gate and α pilot verdicts

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Full run, `logits` (GPU 5, ~10.5 h)

Skip Tasks 7–8 if the Task 6 ruling dropped `logits`; the `logits` result is then its pilot verdict, which Task 8 Step 6 still writes up (report section only, no Q2/Q3).

**Files:** none committed (checkpoints and CSVs live under `runs/`).

**Interfaces:**
- Consumes: the Task 6 ruling (α for `logits`, written `A_LOGITS` below).
- Produces: checkpoints `runs/logs/checkpoints/s2-rule-logits-105k_<timestamp>_/epoch=...-step=step=105000-*.ckpt` (and every 15k), CSV `runs/logs/csv/s2-rule-logits-105k/version_0/metrics.csv`; resolvable by `harness.ckpt_path("s2_rule_logits")`.

- [ ] **Step 1: Launch in the background**

Run: `nvidia-smi --query-gpu=index,memory.used --format=csv -i 5`
Expected: memory used < 1000 MiB; otherwise stop and ask the user.

Run (background): `cd experiments/scale_to_16m_sequences/mcmc_update_rule && GPU_ID=5 ./full.sh "logits A_LOGITS" > logs/full_logits_driver.log 2>&1`
Expected: ~10.5 h. Use a long fallback wake-up (≥ 1 h) and the completion notification; do not poll the log.

- [ ] **Step 2: Verify the run finished**

Run: `tail -3 logs/s2-rule-logits-105k.log; ls runs/logs/checkpoints/ | grep s2-rule-logits`
Expected: the log ends with the trainer's normal exit (no traceback); the run directory holds the `step=step=105000` checkpoint and the 15k–90k checkpoints.

Run: `cd ../../thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -c "import harness; print(harness.ckpt_path('s2_rule_logits'))"`
Expected: one path.

If the run crashed: record it in the ledger with the traceback's last line and ask the user before resuming (resume needs `--resume_data_offset_steps`, as in the 16M S2 run).

---

### Task 8: `logits` evaluation, comparison and report (GPU 5, ~30 min)

**Files (under `experiments/scale_to_16m_sequences/mcmc_update_rule/`):**
- Create: `eval.sh`, `compare.py`, `test_compare.py`, `report/report.md`, `report/compare.md`, `report/figures/ce_vs_k.png`
- Outputs (committed except `.npz`): `results/q2/s2_rule_logits_step105000_noise{3,0}/{config,metrics}.json`, `results/q3/s2_rule_logits_step105000/{config,metrics,metrics_extra}.json`

**Interfaces:**
- Consumes: rule-aware harness and Q3 (Tasks 2, 4); the `logits` checkpoint (Task 7); baseline CSV and results (`experiments/reproduce_xxs_scaling/runs/logs/csv/ebt-s2-xxs-scaling-105k/version_0/metrics.csv`, `experiments/thinking_study/q2_verifier_quality/results/s2_step105000_noise{3,0}/metrics.json`, `experiments/thinking_study/q3_energy_landscape/results/s2_step105000/{metrics,metrics_extra}.json`).
- Produces:
  - `GPU_ID=5 ./eval.sh KEY...` (Q2 noise 3 and 0, Q3, analyzers) — reused by Task 10.
  - `compare.py` with `final_valid_loss(csv) -> float`, `best_k(q3: dict, noise: int) -> tuple[int, float]`, `bon_deltas(q2: dict) -> dict[int, float]`, `gnorm_ratio(q3_extra: dict, noise: int) -> float`; `main()` tabulates every run whose CSV and Q3 results exist, so Task 10 reruns it unchanged.
  - `report/report.md` with a `logits` section — extended by Task 10.

- [ ] **Step 1: Write `eval.sh`**

```bash
#!/usr/bin/env bash
# Q2 (noise 3 and 0, as the baseline) and Q3 for each rule run at 105k; same subsets and seeds as the baseline.
# Usage: GPU_ID=5 ./eval.sh s2_rule_logits s2_rule_log_probs
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$WS/../../thinking_study"
REPO="$(cd "$WS/../../.." && pwd)"
PY=/mnt/data/conda-envs/quangnx/ebt312/bin/python
export HF_HOME=/mnt/data/quangnx/huggingface PYTHONPATH="$REPO:$TS/common" CUDA_VISIBLE_DEVICES="${GPU_ID:?set GPU_ID}"
cd "$REPO"
for model in "$@"; do
  for noise in 3 0; do
    out="$WS/results/q2/${model}_step105000_noise${noise}"
    "$PY" "$TS/q2_verifier_quality/run_q2.py" --model "$model" --noise "$noise" --out-dir "$out"
    "$PY" "$TS/q2_verifier_quality/analyze_q2.py" "$out" > "$out/metrics.log"
  done
  out="$WS/results/q3/${model}_step105000"
  "$PY" "$TS/q3_energy_landscape/run_q3.py" --model "$model" --out-dir "$out"
  "$PY" "$TS/q3_energy_landscape/analyze_q3.py" "$out"
  "$PY" "$TS/q3_energy_landscape/analyze_q3_extra.py" "$out"
done
```

`chmod +x eval.sh`; `bash -n eval.sh` → no output.

- [ ] **Step 2: Run the evaluation in the background**

Run (background): `cd experiments/scale_to_16m_sequences/mcmc_update_rule && GPU_ID=5 ./eval.sh s2_rule_logits > logs/eval_logits.log 2>&1`
Expected: ~30 min; `results/q2/s2_rule_logits_step105000_noise{3,0}/metrics.json` and `results/q3/s2_rule_logits_step105000/{metrics,metrics_extra}.json` exist.

Run: `grep -h mcmc_update_rule results/q3/s2_rule_logits_step105000/config.json`
Expected: `"mcmc_update_rule": "logits"`.

- [ ] **Step 3: Write the failing comparison tests**

`test_compare.py`:

```python
import numpy as np
import pandas as pd

import compare


def test_final_valid_loss_takes_the_last_logged_value(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame({"step": [1, 2, 3, 4], "valid_loss": [np.nan, 4.2, np.nan, 3.9]}).to_csv(path, index=False)
    assert compare.final_valid_loss(path) == 3.9


def test_best_k_and_gnorm_ratio_read_the_q3_fields():
    q3 = {"part_a_noise0": {"best_k": {"k": 5, "ce": 3.8, "ce_at_train_steps": 3.9}}}
    extra = {"part_a_noise3": {"gradient": {"ratio_last_over_trained": {"p1": 0.1, "p50": 0.4, "p99": 2.0}}}}
    assert compare.best_k(q3, 0) == (5, 3.8) and compare.gnorm_ratio(extra, 3) == 0.4


def test_bon_deltas_skip_n1_and_subtract_random():
    q2 = {"e_final": {"best_of_n": {"1": {"ce_energy_pick": 4.0, "ce_random": 4.0},
                                    "2": {"ce_energy_pick": 3.9, "ce_random": 4.0}}}}
    assert compare.bon_deltas(q2) == {2: -0.1}
```

Note `-0.1` compares exactly here because `3.9 - 4.0` is computed the same way in the code; if it does not, round to 6 decimals in `bon_deltas` and in the test.

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cd experiments/scale_to_16m_sequences/mcmc_update_rule && $PY -m pytest test_compare.py -q`
Expected: `ModuleNotFoundError: No module named 'compare'`.

- [ ] **Step 5: Implement `compare.py`, run it**

```python
"""H1-H4 table and CE-vs-k figure: probs baseline (S2 105k/b32) vs the update-rule runs.

Usage: python compare.py   -> report/compare.md, report/figures/ce_vs_k.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent
EXP = WS.parents[1]
TS = EXP / "thinking_study"
RUNS = {  # label -> (metrics.csv, Q2 dir prefix, Q3 dir)
    "probs (baseline)": (EXP / "reproduce_xxs_scaling/runs/logs/csv/ebt-s2-xxs-scaling-105k/version_0/metrics.csv",
                         TS / "q2_verifier_quality/results/s2_step105000", TS / "q3_energy_landscape/results/s2_step105000"),
    "logits": (WS / "runs/logs/csv/s2-rule-logits-105k/version_0/metrics.csv",
               WS / "results/q2/s2_rule_logits_step105000", WS / "results/q3/s2_rule_logits_step105000"),
    "log_probs": (WS / "runs/logs/csv/s2-rule-log-probs-105k/version_0/metrics.csv",
                  WS / "results/q2/s2_rule_log_probs_step105000", WS / "results/q3/s2_rule_log_probs_step105000"),
}


def final_valid_loss(csv) -> float:
    return float(pd.read_csv(csv)["valid_loss"].dropna().iloc[-1])


def best_k(q3: dict, noise: int) -> tuple[int, float]:
    b = q3[f"part_a_noise{noise}"]["best_k"]
    return int(b["k"]), float(b["ce"])


def bon_deltas(q2: dict) -> dict[int, float]:
    return {int(n): v["ce_energy_pick"] - v["ce_random"] for n, v in q2["e_final"]["best_of_n"].items() if n != "1"}


def gnorm_ratio(q3_extra: dict, noise: int) -> float:
    return float(q3_extra[f"part_a_noise{noise}"]["gradient"]["ratio_last_over_trained"]["p50"])


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main():
    present = {k: v for k, v in RUNS.items() if v[0].exists() and (v[2] / "metrics.json").exists()}
    lines = ["| Rule | H1 valid loss @105k | H2 best k (CE), noise 0 | H2 best k (CE), noise 3 | H3 energy − random, N = 2 / 4 / 8 / 16 / 32 (noise 3) | H4 median ‖d₃₁‖/‖d₂‖, noise 0 / 3 |",
             "|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for label, (csv, q2_prefix, q3_dir) in present.items():
        q3, extra = load(q3_dir / "metrics.json"), load(q3_dir / "metrics_extra.json")
        d = bon_deltas(load(Path(f"{q2_prefix}_noise3") / "metrics.json"))
        k0, k3 = best_k(q3, 0), best_k(q3, 3)
        lines.append(f"| {label} | {final_valid_loss(csv):.4f} | {k0[0]} ({k0[1]:.4f}) | {k3[0]} ({k3[1]:.4f}) | "
                     + " / ".join(f"{d[n]:+.3f}" for n in sorted(d)) + f" | {gnorm_ratio(extra, 0):.2f} / {gnorm_ratio(extra, 3):.2f} |")
        for ax, noise in zip(axes, (0, 3)):
            ce = q3[f"part_a_noise{noise}"]["ce_after_k_steps"]
            ks = sorted(map(int, ce))
            ax.plot(ks, [ce[str(k)] for k in ks], marker=".", label=label)
    for ax, noise in zip(axes, (0, 3)):
        ax.axvline(3, color="#8a8984", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("MCMC steps k")
        ax.set_title(f"rp_val CE after k steps (noise {noise})", fontsize=10)
        ax.grid(color="#e6e5e1", lw=0.6)
    axes[0].set_ylabel("CE (nats)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    (WS / "report/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(WS / "report/figures/ce_vs_k.png", dpi=160)
    (WS / "report/compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

Run: `$PY -m pytest test_compare.py -q && $PY compare.py`
Expected: `3 passed`; a table with a `probs (baseline)` row reading `3.9431 | 3 (3.9434) | ... | -0.010 / +0.006 / +0.035 / +0.068 / +0.102 | 0.94 / 0.91` and a `logits` row; `log_probs` is absent because its results do not exist yet.

- [ ] **Step 6: Write `report/report.md` (logits part)**

Structure (Obsidian markdown, 3-bullet TL;DR on top; numbers only from `report/compare.md`, `results/*.json` and the CSVs):
1. TL;DR — about `logits` vs `probs` only; say that `log_probs` is pending.
2. Setup: link the spec; the Task 6 ruling (α per rule) with `results/pilot.json` numbers; what is identical to the baseline.
3. `## logits vs probs`: the H1–H4 table from `report/compare.md`, `![[figures/ce_vs_k.png]]`, one verdict line per hypothesis (supported / not supported / inconclusive), with the single-seed caveat on H1 differences of a few hundredths of a nat. If `logits` was dropped at the pilot, this section reports the pilot numbers instead.
4. `## log_probs vs probs`: one line, "pending (Tasks 9–10)".
5. Reproduction: the commands of Tasks 6–8.

- [ ] **Step 7: Commit**

```bash
cd experiments/scale_to_16m_sequences/mcmc_update_rule
git add eval.sh compare.py test_compare.py report/report.md report/compare.md
git add -f report/figures/ce_vs_k.png results/q2/s2_rule_logits_*/config.json results/q2/s2_rule_logits_*/metrics.json results/q2/s2_rule_logits_*/metrics.log results/q3/s2_rule_logits_*/config.json results/q3/s2_rule_logits_*/metrics.json results/q3/s2_rule_logits_*/metrics_extra.json
git commit -m "report: logits update rule vs the probs baseline (Q2/Q3, H1-H4)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 8: Report to the user before Task 9**

Summarize the `logits` verdicts (H1–H4) to the user, then continue with Task 9. This is a progress note, not a stop: the user already approved the `log_probs` run.

---

### Task 9: Full run, `log_probs` (GPU 5, ~10.5 h)

Skip Tasks 9–10 if the Task 6 ruling dropped `log_probs` (after the α = 1 retry); Task 10 Step 3 then writes up its pilot verdict.

**Files:** none committed.

**Interfaces:**
- Consumes: the Task 6 ruling (α for `log_probs`, written `A_LOG_PROBS` below).
- Produces: checkpoints `runs/logs/checkpoints/s2-rule-log-probs-105k_<timestamp>_/...step=step=105000-*.ckpt` (and every 15k), CSV `runs/logs/csv/s2-rule-log-probs-105k/version_0/metrics.csv`; resolvable by `harness.ckpt_path("s2_rule_log_probs")`.

- [ ] **Step 1: Launch in the background**

Run: `nvidia-smi --query-gpu=index,memory.used --format=csv -i 5`
Expected: memory used < 1000 MiB; otherwise stop and ask the user.

Run (background): `cd experiments/scale_to_16m_sequences/mcmc_update_rule && GPU_ID=5 ./full.sh "log_probs A_LOG_PROBS" > logs/full_log_probs_driver.log 2>&1`
Expected: ~10.5 h. Long fallback wake-up (≥ 1 h); do not poll.

- [ ] **Step 2: Verify the run finished**

Run: `tail -3 logs/s2-rule-log-probs-105k.log; ls runs/logs/checkpoints/ | grep s2-rule-log-probs`
Expected: normal exit; `step=step=105000` and 15k–90k checkpoints present.

Run: `cd ../../thinking_study && PYTHONPATH=$PWD/../..:$PWD/common $PY -c "import harness; print(harness.ckpt_path('s2_rule_log_probs'))"`
Expected: one path. On a crash: ledger it and ask the user before resuming.

---

### Task 10: `log_probs` evaluation and final report (GPU 5, ~30 min)

**Files:**
- Modify: `report/report.md`, `report/compare.md`, `report/figures/ce_vs_k.png`
- Outputs: `results/q2/s2_rule_log_probs_step105000_noise{3,0}/...`, `results/q3/s2_rule_log_probs_step105000/...`

**Interfaces:**
- Consumes: `eval.sh`, `compare.py` (Task 8); the `log_probs` checkpoint (Task 9).
- Produces: the final report covering all three rules.

- [ ] **Step 1: Run the evaluation in the background**

Run (background): `cd experiments/scale_to_16m_sequences/mcmc_update_rule && GPU_ID=5 ./eval.sh s2_rule_log_probs > logs/eval_log_probs.log 2>&1`
Expected: ~30 min; Q2 and Q3 metrics files exist.

Run: `grep -h mcmc_update_rule results/q3/s2_rule_log_probs_step105000/config.json`
Expected: `"mcmc_update_rule": "log_probs"`.

- [ ] **Step 2: Rerun the comparison**

Run: `$PY -m pytest test_compare.py -q && $PY compare.py`
Expected: `3 passed`; the table now has three rows (the `probs` and `logits` rows unchanged from Task 8), and `ce_vs_k.png` has three curves per panel.

- [ ] **Step 3: Extend `report/report.md`**

- Replace the pending `## log_probs vs probs` line with the same structure as the `logits` section: H1–H4 verdicts from the updated table (or the pilot verdict if `log_probs` was dropped).
- Add `## Conclusion`: compare the three rules; does the update rule explain the Φ-balance and best-k = 3 behaviour of the baseline?
- Rewrite the TL;DR to cover all three rules; add the Tasks 9–10 commands to Reproduction.

- [ ] **Step 4: Commit**

```bash
cd experiments/scale_to_16m_sequences/mcmc_update_rule
git add report/report.md report/compare.md
git add -f report/figures/ce_vs_k.png results/q2/s2_rule_log_probs_*/config.json results/q2/s2_rule_log_probs_*/metrics.json results/q2/s2_rule_log_probs_*/metrics.log results/q3/s2_rule_log_probs_*/config.json results/q3/s2_rule_log_probs_*/metrics.json results/q3/s2_rule_log_probs_*/metrics_extra.json
git commit -m "report: log_probs update rule; final three-rule comparison

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
