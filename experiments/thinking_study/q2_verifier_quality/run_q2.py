"""Q2 data collection: does lower energy mean a better prediction?

For each eval sequence, draws G candidates from the EBT's own inference (different initial noise and
Langevin draws, N = the repo's validation step count) and records per token, per candidate:
  energy of the final prediction E(softmax(z_N)) on the last landscape, energy at the last pre-update
  state (what training logs), log p(target), top-1 correctness, entropy.
Also scores off-trajectory candidates with the same energy: Transformer++ distributions at temperatures
0.5/1/2, one-hot(target), one-hot(Transformer++ argmax), uniform.
Pairwise JS divergence between the first 4 candidates measures candidate diversity.

Raw arrays go to results/<run>/arrays.npz (masked tokens only, flattened); metrics come from analyze_q2.py.

Usage: python q2_verifier_quality/run_q2.py --model s2 --step 105000 --noise 3.0 --out-dir ...
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from harness import (RUN_DIRS, batches, energy, entropy, eval_subset_indices, load_model, run_mcmc, token_logprob,
                     tpp_logits, train_eval_steps, validation_split)

LOGP_FLOOR = float(np.log(1e-12))  # log-prob assigned to one-hot candidates that miss the target
TEMPERATURES = (0.5, 1.0, 2.0)
OFF_TRAJ_NAMES = [f"tpp_T{t}" for t in TEMPERATURES] + ["onehot_target", "onehot_tpp_argmax", "uniform"]
N_DIVERSITY = 4


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a.clamp_min(1e-30)) - torch.log(b.clamp_min(1e-30)))).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(RUN_DIRS), required=True)
    ap.add_argument("--tpp", choices=sorted(RUN_DIRS), default="tpp", help="T++ run for off-trajectory candidates")
    ap.add_argument("--tpp-step", type=int, default=105000)
    ap.add_argument("--step", type=int, default=105000)
    ap.add_argument("--noise", type=float, required=True, help="Langevin std at inference")
    ap.add_argument("--n-candidates", type=int, default=32)
    ap.add_argument("--n-seqs", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, hparams = load_model(args.model, args.step)
    tpp, _ = load_model(args.tpp, args.tpp_step)
    val_ds = validation_split(hparams)
    idx = eval_subset_indices(len(val_ds), args.n_seqs)
    n_steps = train_eval_steps(hparams)
    last_lsc = hparams["mcmc_num_steps"] - 1
    G = args.n_candidates

    rec = {k: [] for k in ["e_final", "e_last_logged", "logp", "top1", "entropy", "js", "off_e", "off_logp",
                           "target", "tpp_logp"]}
    t0 = time.time()
    for b, (inputs, targets, mask) in enumerate(batches(val_ds, idx, args.batch_size)):
        inputs, targets, mask = inputs.cuda(), targets.cuda(), mask.cuda()
        ctx = model.embeddings(inputs).detach()
        cand = {k: [] for k in ["e_final", "e_last_logged", "logp", "top1", "entropy"]}
        keep_p = []
        for g in range(G):
            torch.manual_seed(args.seed * 1_000_003 + b * 1009 + g)
            last = {}
            z = run_mcmc(model, inputs, n_steps, args.noise, on_step=lambda s: last.update(e=s.energy))
            p = F.softmax(z, dim=-1)
            with torch.no_grad():
                cand["e_final"].append(energy(model, ctx, p, last_lsc)[mask])
            cand["e_last_logged"].append(last["e"][mask])
            cand["logp"].append(token_logprob(z, targets)[mask])
            cand["top1"].append((z.argmax(-1) == targets)[mask])
            cand["entropy"].append(entropy(p)[mask])
            if g < N_DIVERSITY:
                keep_p.append(p)
        for k, v in cand.items():
            rec[k].append(torch.stack(v).float().cpu())
        with torch.no_grad():
            rec["js"].append(torch.stack([js_divergence(keep_p[0], q)[mask] for q in keep_p[1:]]).cpu())
            del keep_p
            # off-trajectory candidates scored by the same energy
            lt = tpp_logits(tpp, inputs).float()
            off_e, off_lp = [], []
            for temp in TEMPERATURES:
                q = F.softmax(lt / temp, -1)
                off_e.append(energy(model, ctx, q, last_lsc)[mask])
                off_lp.append(token_logprob(lt / temp, targets)[mask])
            V = lt.shape[-1]
            for ids in (targets, lt.argmax(-1)):
                q = F.one_hot(ids, V).float()
                off_e.append(energy(model, ctx, q, last_lsc)[mask])
                off_lp.append(torch.where(ids == targets, 0.0, LOGP_FLOOR)[mask])
            q = torch.full_like(lt, 1.0 / V)
            off_e.append(energy(model, ctx, q, last_lsc)[mask])
            off_lp.append(torch.full_like(targets, -np.log(V), dtype=torch.float)[mask])
            rec["off_e"].append(torch.stack(off_e).cpu())
            rec["off_logp"].append(torch.stack(off_lp).cpu())
            rec["tpp_logp"].append(token_logprob(lt, targets)[mask].cpu())
            rec["target"].append(targets[mask].cpu())
        if b % 8 == 0:
            print(f"batch {b + 1}/{(len(idx) + args.batch_size - 1) // args.batch_size} "
                  f"{time.time() - t0:.0f}s", flush=True)

    arrays = {k: torch.cat(v, dim=-1).numpy() for k, v in rec.items()}
    np.savez_compressed(args.out_dir / "arrays.npz", **arrays)
    config = vars(args) | {"out_dir": str(args.out_dir), "n_steps": n_steps, "alpha": float(model.alpha),
                           "landscape": last_lsc, "off_trajectory": OFF_TRAJ_NAMES, "n_tokens": int(arrays["logp"].shape[1]),
                           "seconds": round(time.time() - t0, 1)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
