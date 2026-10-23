"""Steady-state DDP step time on GPUs 0-3: (t_350 - t_50) / 300 steps, peak GPU memory, and the ETA for 1M steps.

Usage: GPU_IDS=0,1,2,3 python benchmark.py {s2,tpp} [extra run-script args, e.g. --dataset_dir data/smoke]
Writes logs/benchmark_<model>.json. The ETA excludes validation.
"""
import json
import os
import subprocess
import sys
import threading
import time

WS = os.path.dirname(os.path.abspath(__file__))
PLANNED = {"paper": 1_000_000}


def gpu_memory_mib(ids: str) -> list[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", ids],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


def timed_run(model: str, steps: int, extra: list[str], peak: list[int]) -> float:
    cmd = ["bash", os.path.join(WS, f"run_{model}.sh"), "--run_name", f"bench-{model}-{steps}",
           "--wandb_project", "ebt_scale_paper_bench", "--max_steps", str(steps),
           "--limit_val_batches", "0", "--val_check_interval", "100", "--save_top_k_ckpts", "0", *extra]
    done = threading.Event()

    def poll() -> None:
        while not done.wait(5):
            peak[0] = max(peak[0], *gpu_memory_mib(os.environ["GPU_IDS"]))

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    start = time.perf_counter()
    try:
        with open(os.path.join(WS, "logs", f"bench_{model}_{steps}.log"), "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        done.set()
    return time.perf_counter() - start


def main(model: str, extra: list[str]) -> None:
    peak = [0]
    sps = 300 / (timed_run(model, 350, extra, peak) - timed_run(model, 50, extra, peak))
    result = {"model": model, "gpus": os.environ["GPU_IDS"], "steps_per_s": round(sps, 3), "peak_gpu_mem_mib": peak[0],
              **{f"eta_h_{k}": round(v / sps / 3600, 2) for k, v in PLANNED.items()}, "extra_args": extra}
    with open(os.path.join(WS, "logs", f"benchmark_{model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
