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
