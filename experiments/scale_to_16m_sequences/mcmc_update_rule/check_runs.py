"""Verdicts from Lightning metrics.csv files: the probs gate against the baseline, and the α pilots.

gate:  the refactored probs path reproduces the baseline's train loss at its first logged steps.
pilot: the MCMC loop does work under a new rule: mean (CE after the first step - CE after the last step)
       over logged steps 1,500-2,000 is at least half the baseline's, with no NaN/Inf and the run complete.

Usage: python check_runs.py gate RUN_CSV | python check_runs.py pilot RUN_CSV [RUN_CSV ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BASELINE_CSV = REPO / "experiments/reproduce_xxs_scaling/runs/logs/csv/ebt-s2-xxs-scaling-105k/version_0/metrics.csv"
GATE_STEPS, GATE_RTOL = (49, 99, 149), 0.02
WINDOW, LAST_STEP, MIN_GAIN_RATIO = (1500, 2000), 1999, 0.5


def train_rows(csv) -> pd.DataFrame:
    d = pd.read_csv(csv)
    return d[d["train_loss"].notna()].set_index("step")


def gate(run, base=BASELINE_CSV) -> dict:
    r, b = train_rows(run), train_rows(base)
    steps = {s: (float(r.at[s, "train_loss"]), float(b.at[s, "train_loss"])) for s in GATE_STEPS if s in r.index}
    ok = len(steps) == len(GATE_STEPS) and all(abs(x - y) <= GATE_RTOL * abs(y) for x, y in steps.values())
    return {"steps": steps, "passed": ok}


def mcmc_gain(rows: pd.DataFrame) -> float:
    w = rows[(rows.index >= WINDOW[0]) & (rows.index <= WINDOW[1])]
    return float((w["train_initial_loss"] - w["train_final_step_loss"]).mean())


def pilot_verdict(run, base=BASELINE_CSV) -> dict:
    raw = pd.read_csv(run)
    logged = raw[raw["train_initial_loss"].notna()]
    finite = bool(np.isfinite(logged[["train_loss", "train_final_step_loss"]].to_numpy()).all())
    rows, b = train_rows(run), train_rows(base)
    completed = bool(len(rows)) and int(rows.index.max()) >= LAST_STEP
    gain, base_gain = mcmc_gain(rows), mcmc_gain(b)
    w = rows[(rows.index >= WINDOW[0]) & (rows.index <= WINDOW[1])]
    return {"finite": finite, "completed": completed, "gain": gain, "baseline_gain": base_gain,
            "final_step_loss": float(w["train_final_step_loss"].mean()),
            "passed": finite and completed and gain >= MIN_GAIN_RATIO * base_gain}


def main():
    cmd, paths = sys.argv[1], sys.argv[2:]
    if cmd == "gate":
        v = gate(paths[0])
        print(json.dumps(v, indent=1))
        sys.exit(0 if v["passed"] else 1)
    print(json.dumps({p: pilot_verdict(p) for p in paths}, indent=1))


if __name__ == "__main__":
    main()
