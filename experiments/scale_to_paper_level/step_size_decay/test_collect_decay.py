import pytest

from collect_decay import drift, select_d_star, verdict_d1


def test_select_d_star_excludes_ema():
    # rp_val CE at k = 3; the EMA combination is lowest but is not a candidate for d*
    rp3 = {"gd_a2.5": 3.60, "gd_a5_harm": 3.50, "gd_a7.5_g0.7": 3.55, "ema_m0.5_harm": 3.40}
    assert select_d_star(rp3) == "gd_a5_harm"


def test_verdict_d1_bands():
    assert verdict_d1(72.0) == "step size explains most"
    assert verdict_d1(73.2) == "step size explains most"
    assert verdict_d1(75.0) == "partial"
    assert verdict_d1(78.0) == "momentum does more"
    assert verdict_d1(90.0) == "momentum does more"


def test_drift():
    assert drift(72.2, 78.4) == pytest.approx(78.4 / 72.2 - 1)
    assert drift(80.0, 80.0) == 0.0
