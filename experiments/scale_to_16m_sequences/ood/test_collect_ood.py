import math

import collect_ood


def test_pct_improvement_matches_perplexity_ratio():
    assert collect_ood.pct_improvement(2.0, 2.0) == 0.0
    assert math.isclose(collect_ood.pct_improvement(2.0, 2.0 - math.log(2)), 50.0)
    assert collect_ood.pct_improvement(2.0, 2.1) < 0


def test_rq1_row_picks_the_best_n():
    bon = {str(n): {"ce_energy_pick": 3.0 + d, "ce_random": 3.0} for n, d in [(2, -0.03), (4, -0.01), (8, 0.02)]}
    row = collect_ood.rq1_row({"e_final": {"best_of_n": bon}})
    assert row["best_n"] == 2 and math.isclose(row["best_delta"], -0.03)


def test_rq2_row_uses_trained_k_as_default_and_best_longer_k():
    ce = {"1": 6.0, "2": 4.0, "3": 3.5, "4": 3.6, "5": 3.45, "6": 3.55}
    row = collect_ood.rq2_row({"part_a_noise0": {"ce_after_k_steps": ce, "best_k": {"k": 5}}}, 0)
    assert row["ce_k3"] == 3.5 and row["best_longer_k"] == 5 and math.isclose(row["best_longer_delta"], -0.05)


def test_rq1_row_ignores_n1_which_is_no_selection():
    bon = {str(n): {"ce_energy_pick": 3.0 + d, "ce_random": 3.0} for n, d in [(1, 0.0), (2, 0.01), (4, 0.05)]}
    row = collect_ood.rq1_row({"e_final": {"best_of_n": bon}})
    assert row["best_n"] == 2 and math.isclose(row["best_delta"], 0.01)
