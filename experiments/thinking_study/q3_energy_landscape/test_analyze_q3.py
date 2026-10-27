import numpy as np

import analyze_q3
import analyze_q3_extra


def test_share_is_none_when_rho_is_undefined():
    # non-probs rules record rho as NaN; "0% contracting" would be a false reading
    rho = np.full(8, np.nan)
    assert analyze_q3.share_or_none(rho < 1, rho) is None
    assert analyze_q3.share_or_none(np.array([0.5, 2.0]) < 1, np.array([0.5, 2.0])) == 0.5


def test_hessian_metrics_report_no_rho_share_for_nan_rho():
    ev = np.sort(np.random.default_rng(0).normal(size=(4, 6)), axis=1)
    out = analyze_q3_extra.hessian_metrics(ev, np.full(4, np.nan), alpha=5.0)
    assert out["share_rho_above_1"] is None
