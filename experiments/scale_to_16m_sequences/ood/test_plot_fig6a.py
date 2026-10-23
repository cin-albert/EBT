import numpy as np

import plot_fig6a


def test_bon_ce_takes_the_lowest_energy_among_the_first_n_candidates():
    e = np.array([[1.0, 5.0], [0.5, 6.0], [0.1, 0.1]])  # (candidates, tokens)
    ce = np.array([[2.0, 3.0], [4.0, 1.0], [9.0, 9.0]])
    assert plot_fig6a.bon_ce(e, ce, 2) == (4.0 + 3.0) / 2  # candidate 3 is outside N = 2


def test_mean_ppl_averages_perplexities_not_ce():
    assert plot_fig6a.mean_ppl([0.0, np.log(3.0)]) == 2.0
