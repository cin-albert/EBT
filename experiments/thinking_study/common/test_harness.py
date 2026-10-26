import numpy as np
import pytest
import torch
import torch.nn.functional as F

import harness
from fake_ebt import FakeEBT
from model.nlp.mcmc_update import RULES, next_logits


def test_ckpt_path_finds_any_epoch_and_new_run_keys(tmp_path, monkeypatch):
    run = tmp_path / "s2-b64-16m_2026-10-17_00-00-00_"
    run.mkdir()
    (run / "epoch=epoch=1-step=step=250000-valid_loss=valid_loss=3.9000.ckpt").touch()
    monkeypatch.setitem(harness.RUN_DIRS, "s2_b64_16m", (tmp_path, "s2-b64-16m_20*"))
    assert harness.ckpt_path("s2_b64_16m", 250000).name.startswith("epoch=epoch=1-step=step=250000")


def test_run_glob_does_not_match_longer_run_name(tmp_path, monkeypatch):
    for name in ("s2-b64-16m_2026-10-17_00-00-00_", "s2-b64-16m-seed2_2026-10-17_00-00-00_"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "epoch=epoch=0-step=step=25000-valid_loss=valid_loss=4.0.ckpt").touch()
    monkeypatch.setitem(harness.RUN_DIRS, "s2_b64_16m", (tmp_path, "s2-b64-16m_20*"))
    assert "seed2" not in str(harness.ckpt_path("s2_b64_16m", 25000))


def test_validation_split_uses_fixed_indices_when_present(tmp_path, monkeypatch):
    np.save(tmp_path / "val.npy", np.array([4, 0, 2], dtype=np.int64))
    monkeypatch.setattr("data.nlp.pajama_subset.RedPajamaSubsetDataset", lambda hp: list(range(6)))
    val = harness.validation_split({"fixed_val_indices": str(tmp_path / "val.npy"), "validation_split_pct": 0.5})
    assert [val[i] for i in range(len(val))] == [4, 0, 2]


def test_validation_split_falls_back_to_seeded_random_split(monkeypatch):
    monkeypatch.setattr("data.nlp.pajama_subset.RedPajamaSubsetDataset", lambda hp: list(range(100)))
    a = harness.validation_split({"validation_split_pct": 0.1})
    b = harness.validation_split({"validation_split_pct": 0.1})
    assert len(a) == 10 and list(a.indices) == list(b.indices)


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
