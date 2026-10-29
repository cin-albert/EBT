"""Per-step arrays of one momentum arm on one eval set (Q3 Part A fields, plus seq_id).

Per token and step t (state before the step's update): energy, entropy of p_t, log p(target) under softmax(z_t),
TV(p_{t+1}, p_t) (the last one against softmax(z_N)); plus log p(target) after the last step. Same sequences,
batching and seeds as Q3 Part A (torch.manual_seed(seed + b)), so arm gd_a5 reproduces ../results/q3.
Writes part_a_noise{0,3}.npz and config.json (written last; run_all.sh skips a run that has it).

Usage: python run_momentum.py --arm ema_m0.5 --data gsm8k --out-dir results/single/ema_m0.5_gsm8k
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from harness import RUN_DIRS, batches, entropy, eval_dataset, eval_subset_indices, load_model, token_logprob
from momentum import ARMS, Arm, momentum_mcmc

FIELDS = ("energy", "entropy", "logp", "tv_next")


def record(model, ds, idx: list[int], n_steps: int, noise: float, arm: Arm, batch_size: int, seed: int,
           device: str = "cuda") -> dict[str, np.ndarray]:
    rec = {k: [] for k in (*FIELDS, "logp_final", "seq_id")}
    for b, (inputs, targets, mask) in enumerate(batches(ds, idx, batch_size)):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        torch.manual_seed(seed + b)
        per = {k: [] for k in FIELDS}
        gen = momentum_mcmc(model, inputs, n_steps, noise, arm)
        prev_p = None
        while True:
            try:
                s = next(gen)
            except StopIteration as stop:
                z_final = stop.value
                break
            per["energy"].append(s.energy[mask])
            per["entropy"].append(entropy(s.p)[mask])
            per["logp"].append(token_logprob(s.z, targets)[mask])
            if prev_p is not None:
                per["tv_next"].append((0.5 * (s.p - prev_p).abs().sum(-1))[mask])
            prev_p = s.p
        per["tv_next"].append((0.5 * (F.softmax(z_final, -1) - prev_p).abs().sum(-1))[mask])
        for k, v in per.items():
            rec[k].append(torch.stack(v).float().cpu())
        rec["logp_final"].append(token_logprob(z_final, targets)[mask].float().cpu())
        rows = torch.arange(b * batch_size, b * batch_size + len(inputs), device=device)
        rec["seq_id"].append(rows[:, None].expand_as(mask)[mask].cpu())
    return {k: torch.cat(v, dim=-1).numpy() for k, v in rec.items()}


def save_record(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--data", required=True, help="rp_val or an OOD set (ood_data.OOD_SETS)")
    ap.add_argument("--model", choices=sorted(RUN_DIRS), default="s2_paper")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--n-seqs", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arm = ARMS[args.arm]
    model, hparams = load_model(args.model, args.step)
    ds = eval_dataset(args.data, hparams)
    idx = eval_subset_indices(len(ds), args.n_seqs)
    noises = sorted({0.0, float(hparams["langevin_dynamics_noise"])})
    t0 = time.time()
    for noise in noises:
        arrays = record(model, ds, idx, args.n_steps, noise, arm, args.batch_size, args.seed)
        save_record(args.out_dir / f"part_a_noise{noise:g}.npz", arrays)
        print(f"{args.arm} {args.data} noise={noise:g}: CE after k=3 {-arrays['logp'][3].mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    config = vars(args) | {"out_dir": str(args.out_dir), "rule": arm.rule, "mu": arm.mu, "alpha": arm.alpha,
                           "noises": noises, "n_seqs_used": len(idx), "seconds": round(time.time() - t0, 1)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
