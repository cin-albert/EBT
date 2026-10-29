import math

import numpy as np
import pytest

from collect_momentum import (bon_ce, ce_after_k, mean_ood_ppl, paired_bootstrap, phase3_gate, select, verdict_m1)


def test_ce_after_k_convention():
    a = {"logp": -np.arange(12, dtype=float).reshape(4, 3), "logp_final": np.array([-7.0, -8.0, -9.0])}
    assert np.array_equal(ce_after_k(a, 2), [6.0, 7.0, 8.0])
    assert np.array_equal(ce_after_k(a, 4), [7.0, 8.0, 9.0])
    with pytest.raises(ValueError):
        ce_after_k(a, 5)


def test_mean_ood_ppl():
    ce = {"a": math.log(10), "b": math.log(30)}
    assert mean_ood_ppl(ce) == pytest.approx(20.0)
    assert mean_ood_ppl(ce, geometric=True) == pytest.approx(math.sqrt(300))


def test_paired_bootstrap_sign():
    rng = np.random.default_rng(0)
    seq_id = np.repeat(np.arange(50), 4)
    ce_b = rng.normal(4.0, 1.0, seq_id.size)
    mean, lo, hi = paired_bootstrap(ce_b - 0.1, ce_b, seq_id)
    assert mean == pytest.approx(-0.1) and lo == pytest.approx(-0.1) and hi == pytest.approx(-0.1) and hi < 0


def test_paired_bootstrap_ci_straddles_zero_for_noise():
    rng = np.random.default_rng(1)
    seq_id = np.repeat(np.arange(50), 4)
    _, lo, hi = paired_bootstrap(rng.normal(0, 1, 200), rng.normal(0, 1, 200), seq_id)
    assert lo < 0 < hi


def test_selection_uses_rp_val_only():
    # ce[arm][set][k]; hb_m0.9 is best on OOD but ema_m0.5 is best on rp_val at k = 3
    ce = {
        "gd_a5": {"rp_val": {3: 3.6, 5: 3.7, 8: 3.8, 10: 3.9}, "gsm8k": {3: 3.9}},
        "gd_a10": {"rp_val": {3: 3.5, 5: 3.6, 8: 3.7, 10: 3.8}, "gsm8k": {3: 3.0}},
        "hb_m0.9": {"rp_val": {3: 3.55, 5: 3.4, 8: 3.3, 10: 3.2}, "gsm8k": {3: 2.0}},
        "ema_m0.5": {"rp_val": {3: 3.45, 5: 3.44, 8: 3.40, 10: 3.42}, "gsm8k": {3: 3.9}},
    }
    assert select(ce) == ("ema_m0.5", "gd_a10", 8)


def test_m1_needs_three_sets():
    assert not verdict_m1(3.0, [-0.01, -0.02, 0.01, 0.02])
    assert verdict_m1(3.0, [-0.01, -0.02, -0.03, 0.02])
    assert not verdict_m1(1.5, [-0.01, -0.02, -0.03, -0.01])


def test_gate_flag():
    assert phase3_gate({"M1": False, "M3": False, "M2": True})
    assert not phase3_gate({"M1": True, "M3": False})
    assert not phase3_gate({"M1": False, "M3": True})


def test_gate_ignores_pending_m5():
    assert phase3_gate({"M1": True, "M3": False, "M5": None}) is False
    assert phase3_gate({"M1": False, "M3": False, "M5": None}) is True


def test_bon_ce_picks_lowest_energy_among_first_n():
    e = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    logp = np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]])
    assert bon_ce(e, logp, 1) == pytest.approx(1.5)   # candidate 0
    assert bon_ce(e, logp, 2) == pytest.approx((3 + 2) / 2)
    assert bon_ce(e, logp, 3) == pytest.approx((5 + 6) / 2)
