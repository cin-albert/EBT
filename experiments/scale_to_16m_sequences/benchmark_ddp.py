"""Steady-state DDP step time: (t_350 - t_50) / 300 steps, then ETAs for the planned step counts."""
import json
import os
import subprocess
import sys
import time

WS = os.path.dirname(os.path.abspath(__file__))
PLANNED = {"16m": 250_000}


def timed_run(model: str, steps: int, extra: list[str]) -> float:
    cmd = ["bash", os.path.join(WS, f"run_{model}.sh"), "--run_name", f"bench-{model}-{steps}",
           "--wandb_project", "ebt_scale16m_bench", "--max_steps", str(steps),
           "--limit_val_batches", "0", "--val_check_interval", "100", "--save_top_k_ckpts", "0", *extra]
    start = time.perf_counter()
    with open(os.path.join(WS, "logs", f"bench_{model}_{steps}.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def main(model: str, extra: list[str]) -> None:
    sps = 300 / (timed_run(model, 350, extra) - timed_run(model, 50, extra))
    result = {"model": model, "steps_per_s": round(sps, 3), **{f"eta_h_{k}": round(v / sps / 3600, 2) for k, v in PLANNED.items()}}
    with open(os.path.join(WS, "logs", f"benchmark_{model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])  # extra args are passed to the run script (e.g. --dataset_dir)
