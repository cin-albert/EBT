"""Measure steady-state step time and peak GPU memory for the xxs runs; prints a 105k-step ETA."""
import json
import os
import subprocess
import sys
import threading
import time

EXP = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(EXP, "logs")
RUNS_DIR = os.path.join(EXP, "runs")
TOTAL_STEPS = 105000


def timed_run(model, steps):
    cmd = ["bash", os.path.join(EXP, f"run_{model}_xxs.sh"),
           "--run_name", f"bench-{model}-{steps}", "--wandb_project", "ebt_xxs_bench",
           "--max_steps", str(steps), "--limit_val_batches", "0",
           # the smoke subset has ~610 train batches; the default 15000 would fail Lightning's interval check
           "--val_check_interval", "100"]
    start = time.perf_counter()
    with open(os.path.join(LOG_DIR, f"bench_{model}_{steps}.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - start


def poll_memory(gpu_id, samples, stop):
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", gpu_id],
                             capture_output=True, text=True).stdout.strip()
        if out:
            samples.append(int(out))
        stop.wait(1.0)


def main(model):
    gpu_id = os.environ["GPU_ID"]
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    t_short = timed_run(model, 50)
    samples, stop = [], threading.Event()
    poller = threading.Thread(target=poll_memory, args=(gpu_id, samples, stop), daemon=True)
    poller.start()
    t_long = timed_run(model, 350)
    stop.set()
    poller.join()
    steps_per_s = 300 / (t_long - t_short)
    result = {
        "model": model,
        "seconds_50_steps": round(t_short, 1),
        "seconds_350_steps": round(t_long, 1),
        "steps_per_s": round(steps_per_s, 3),
        "tokens_per_s": round(steps_per_s * 32 * 256),
        "peak_mem_mib": max(samples) if samples else None,
        "eta_hours_105k": round(TOTAL_STEPS / steps_per_s / 3600, 2),
    }
    with open(os.path.join(RUNS_DIR, f"benchmark_{model}.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
