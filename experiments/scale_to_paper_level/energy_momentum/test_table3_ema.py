from pathlib import Path

import numpy as np
import pytest
import torch

import harness
from momentum import ARMS
from table3_ema import ppl_ratio_ci, score
from test_momentum import stub_model
from test_run_momentum import toy_dataset

WS = Path(__file__).resolve().parent
PROTOCOL = WS.parent / "results/table3_protocol"


def test_score_gd_a5_matches_protocol_draws():
    # table3_protocol.score: torch.manual_seed(b), run_mcmc at the trained α, batch 32 in dataset order
    model, ds = stub_model(), toy_dataset()
    r = score(model, ds, ARMS["gd_a5"], n_steps=3, noise=3.0, batch_size=2, device="cpu")
    ref_ce, ref_n = [], []
    for b, (inputs, targets, mask) in enumerate(harness.batches(ds, list(range(len(ds))), 2)):
        torch.manual_seed(b)
        nll = -harness.token_logprob(harness.run_mcmc(model, inputs, 3, 3.0), targets) * mask
        ref_ce.append(nll.sum(1).double().numpy())
        ref_n.append(mask.sum(1).numpy())
    assert np.array_equal(r["ce_sum"], np.concatenate(ref_ce))
    assert np.array_equal(r["n_tok"], np.concatenate(ref_n))


def test_score_ema_differs_from_gd():
    model, ds = stub_model(), toy_dataset()
    gd = score(model, ds, ARMS["gd_a5"], n_steps=3, noise=3.0, batch_size=2, device="cpu")
    ema = score(model, ds, ARMS["ema_m0.5"], n_steps=3, noise=3.0, batch_size=2, device="cpu")
    assert np.array_equal(gd["n_tok"], ema["n_tok"]) and not np.allclose(gd["ce_sum"], ema["ce_sum"])


def test_ppl_ratio_ci_constant_shift():
    rng = np.random.default_rng(0)
    n_tok = rng.integers(5, 20, 100)
    ce_b = rng.normal(4.0, 1.0, 100) * n_tok
    ratio, lo, hi = ppl_ratio_ci(ce_b - 0.1 * n_tok, ce_b, n_tok)  # 0.1 nat per token lower
    assert ratio == pytest.approx(np.exp(-0.1)) and lo == pytest.approx(np.exp(-0.1)) and hi == pytest.approx(np.exp(-0.1))


def test_ppl_ratio_ci_straddles_one_for_noise():
    rng = np.random.default_rng(1)
    n_tok = np.full(200, 10)
    _, lo, hi = ppl_ratio_ci(rng.normal(40, 5, 200), rng.normal(40, 5, 200), n_tok)
    assert lo < 1 < hi


@pytest.mark.skipif(not torch.cuda.is_available() or not (PROTOCOL / "s2_paper_bigbench_dyck_languages.npz").exists(),
                    reason="needs a GPU and the table3_protocol outputs")
def test_gd_a5_reproduces_table3_protocol_s2_on_dyck():
    model, hp = harness.load_model("s2_paper", 1_000_000)
    r = score(model, harness.eval_dataset("bigbench_dyck_languages", hp), ARMS["gd_a5"], n_steps=3, noise=3.0)
    ref = np.load(PROTOCOL / "s2_paper_bigbench_dyck_languages.npz")
    assert np.array_equal(r["n_tok"], ref["n_tok"])
    assert np.allclose(r["ce_sum"], ref["ce_sum"], rtol=0, atol=1e-6)
