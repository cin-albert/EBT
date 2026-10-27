import numpy as np
import pandas as pd

import compare


def test_valid_loss_at_takes_the_value_logged_just_before_the_checkpoint_step(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame({"step": [100, 14999, 20000, 29999, 104999],
                  "valid_loss": [np.nan, 5.5, np.nan, 5.4, 18.0]}).to_csv(path, index=False)
    assert compare.valid_loss_at(path, 30000) == 5.4
    assert compare.valid_loss_at(path, 105000) == 18.0


def test_best_k_and_gnorm_ratio_read_the_q3_fields():
    q3 = {"part_a_noise0": {"best_k": {"k": 5, "ce": 3.8, "ce_at_train_steps": 3.9}}}
    extra = {"part_a_noise3": {"gradient": {"ratio_last_over_trained": {"p1": 0.1, "p50": 0.4, "p99": 2.0}}}}
    assert compare.best_k(q3, 0) == (5, 3.8) and compare.gnorm_ratio(extra, 3) == 0.4


def test_bon_deltas_skip_n1_and_subtract_random():
    q2 = {"e_final": {"best_of_n": {"1": {"ce_energy_pick": 4.0, "ce_random": 4.0},
                                    "2": {"ce_energy_pick": 3.9, "ce_random": 4.0}}}}
    assert compare.bon_deltas(q2) == {2: -0.1}


def test_missing_noise_level_reads_as_none():
    assert compare.best_k({"part_a_noise0": {}}, 3) is None
    assert compare.gnorm_ratio({"part_a_noise0": {}}, 3) is None
