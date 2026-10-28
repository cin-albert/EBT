import numpy as np
import pytest

from collect_paper import bon_ce, pct_improvement, spearman, val_curve, verdicts


def fake_results(gap=0.10, mean_ppl_k2=100.0, mean_ppl_k3=80.0, mean_ppl_bon10=70.0,
                 dyck_pct=(1.0, 2.0, 3.0), val_pct=(0.5, 1.0, 1.5), shift_pct=((1.0, 5.0), (2.0, 10.0), (3.0, 15.0)),
                 ood_ppl=((30.0, 31.0), (30.0, 31.0), (30.0, 31.0), (30.0, 29.0))) -> dict:
    """Synthetic collect_paper results; every claim is Supported with the defaults."""
    tokens = [1e9, 2e9, 3e9]
    return {"gap": gap, "mean_ppl": {"k2": mean_ppl_k2, "k3": mean_ppl_k3, "bon10": mean_ppl_bon10},
            "dyck_curve": dict(zip(tokens, dyck_pct)), "val_curve": dict(zip(tokens, val_pct)),
            "shift_points": list(shift_pct), "ood_ppl": dict(zip("abcd", ood_ppl))}


def row(rows, claim):
    return next(r for r in rows if r["claim"] == claim)


def test_bon_ce_picks_lowest_energy():
    e = np.array([[1.0, 0.0], [0.0, 1.0]]); ce = np.array([[5.0, 1.0], [2.0, 9.0]])
    assert bon_ce(e, ce, 2) == 1.5


def test_bon_ce_uses_only_first_n():
    e = np.array([[1.0], [0.0]]); ce = np.array([[5.0], [2.0]])
    assert bon_ce(e, ce, 1) == 5.0


def test_pct_improvement():
    assert pct_improvement(4.0, 4.0) == 0.0 and pct_improvement(4.0, 3.9) > 0
    assert pct_improvement(np.log(100), np.log(89)) == pytest.approx(11.0)


def test_spearman_sign():
    assert spearman([1, 2, 3], [1, 4, 9]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_all_supported_with_defaults():
    assert [r["verdict"].startswith("Supported") for r in verdicts(fake_results())] == [True] * 7


def test_verdict_c1_thresholds():
    assert row(verdicts(fake_results(gap=0.114)), "C1")["verdict"] == "Supported (matches paper)"
    assert row(verdicts(fake_results(gap=0.148)), "C1")["verdict"] == "Supported"
    assert row(verdicts(fake_results(gap=0.150)), "C1")["verdict"] == "Not supported"


def test_verdict_c2_threshold():
    rows = verdicts(fake_results(mean_ppl_k2=100.0, mean_ppl_k3=89.0))
    assert row(rows, "C2")["verdict"] == "Supported"     # 11% > 10%
    rows = verdicts(fake_results(mean_ppl_k2=100.0, mean_ppl_k3=91.0))
    assert row(rows, "C2")["verdict"] == "Not supported"


def test_verdict_c3_threshold():
    assert row(verdicts(fake_results(mean_ppl_k3=80.0, mean_ppl_bon10=78.0)), "C3")["verdict"] == "Supported"  # 2.5%
    assert row(verdicts(fake_results(mean_ppl_k3=80.0, mean_ppl_bon10=78.8)), "C3")["verdict"] == "Not supported"  # 1.5%


def test_verdict_c4_needs_positive_final_and_trend():
    assert row(verdicts(fake_results(dyck_pct=(3.0, 2.0, 1.0))), "C4")["verdict"] == "Not supported"  # falling trend
    assert row(verdicts(fake_results(dyck_pct=(-3.0, -2.0, -0.1))), "C4")["verdict"] == "Not supported"  # final <= 0


def test_verdict_c5_needs_positive_final_and_trend():
    assert row(verdicts(fake_results(val_pct=(1.5, 1.0, 0.5))), "C5")["verdict"] == "Not supported"
    assert row(verdicts(fake_results(val_pct=(-1.5, -1.0, -0.5))), "C5")["verdict"] == "Not supported"


def test_verdict_c6_sign():
    falling = ((1.0, 15.0), (2.0, 10.0), (3.0, 5.0))
    assert row(verdicts(fake_results(shift_pct=falling)), "C6")["verdict"] == "Not supported"


def test_verdict_c7_three_of_four():
    two = ((30.0, 31.0), (30.0, 31.0), (30.0, 29.0), (30.0, 29.0))
    assert row(verdicts(fake_results(ood_ppl=two)), "C7")["verdict"] == "Not supported"


def test_val_curve_keeps_the_first_row_per_step(tmp_path):
    # a resumed run repeats the checkpoint's validation on 1 batch; the original row is the real one
    for v, rows in [(0, [(49999, 3.7), (99999, 3.6)]), (1, [(99999, 3.9), (149999, 3.5)])]:
        d = tmp_path / f"version_{v}"; d.mkdir()
        (d / "metrics.csv").write_text("step,valid_loss,train_loss\n" + "".join(f"{s},{l},\n{s},,1.0\n" for s, l in rows))
    assert val_curve(tmp_path, "valid_loss") == {50000: 3.7, 100000: 3.6, 150000: 3.5}
