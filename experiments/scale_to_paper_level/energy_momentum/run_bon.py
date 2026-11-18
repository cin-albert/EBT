"""Self-verification (BoN) candidates of one momentum arm: G trajectories of k steps per sequence.

Per candidate and token: energy of the final prediction E(softmax(z_k)) on the last landscape (Q2's e_final)
and log p(target). Seeds follow Q2: torch.manual_seed(seed * 1_000_003 + b * 1009 + g).
Writes arrays.npz (e_final, logp: (G, tokens); seq_id: (tokens,)) and config.json.

Usage: python run_bon.py --arm ema_m0.5 --k 3 --data gsm8k --out-dir results/bon/ema_m0.5_k3_gsm8k
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from harness import RUN_DIRS, batches, energy, eval_dataset, eval_subset_indices, load_model, token_logprob
from momentum import ARMS, Arm, run_momentum_mcmc


def record_bon(model, ds, idx: list[int], arm: Arm, k: int, n_candidates: int, noise: float, batch_size: int,
               seed: int, device: str = "cuda") -> dict[str, np.ndarray]:
    last_lsc = model.hparams.mcmc_num_steps - 1
    rec = {"e_final": [], "logp": [], "seq_id": []}
    for b, (inputs, targets, mask) in enumerate(batches(ds, idx, batch_size)):
        inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
        ctx = model.embeddings(inputs).detach()
        e_final, logp = [], []
        for g in range(n_candidates):
            torch.manual_seed(seed * 1_000_003 + b * 1009 + g)
            z = run_momentum_mcmc(model, inputs, k, noise, arm)
            with torch.no_grad():
                e_final.append(energy(model, ctx, F.softmax(z, -1), last_lsc)[mask])
            logp.append(token_logprob(z, targets)[mask])
        rec["e_final"].append(torch.stack(e_final).float().cpu())
        rec["logp"].append(torch.stack(logp).float().cpu())
        rows = torch.arange(b * batch_size, b * batch_size + len(inputs), device=device)
        rec["seq_id"].append(rows[:, None].expand_as(mask)[mask].cpu())
    return {k_: torch.cat(v, dim=-1).numpy() for k_, v in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--k", type=int, required=True, help="MCMC steps per candidate")
    ap.add_argument("--data", required=True, help="rp_val or an OOD set (ood_data.OOD_SETS)")
    ap.add_argument("--model", choices=sorted(RUN_DIRS), default="s2_paper")
    ap.add_argument("--step", type=int, default=1000000)
    ap.add_argument("--n-candidates", type=int, default=10)
    ap.add_argument("--noise", type=float, default=3.0)
    ap.add_argument("--n-seqs", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, hparams = load_model(args.model, args.step)
    ds = eval_dataset(args.data, hparams)
    idx = eval_subset_indices(len(ds), args.n_seqs)
    t0 = time.time()
    arrays = record_bon(model, ds, idx, ARMS[args.arm], args.k, args.n_candidates, args.noise, args.batch_size,
                        args.seed)
    np.savez_compressed(args.out_dir / "arrays.npz", **arrays)
    config = vars(args) | {"out_dir": str(args.out_dir), "n_seqs_used": len(idx), "seconds": round(time.time() - t0, 1)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
