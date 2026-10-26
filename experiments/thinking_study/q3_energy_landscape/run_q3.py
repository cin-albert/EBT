"""Q3 data collection: properties of the learned energy landscape and of the EBT update map.

Mechanism (from model/nlp/ebt.py; see harness docstring): with p_t = softmax(z_t) and g_t = dE/dp at p_t,
the next state is z_{t+1} = p_t - alpha * g_t, so p_{t+1} = T(p_t) := softmax(p_t - alpha * g_t).
Fixed points of T are exactly the stationary points on the simplex of
    Phi(p) = alpha * E(p) - H(p) - 0.5 * ||p||^2
(set the gradient of the Lagrangian to zero: alpha*g + log p + 1 - p = const). So inference is a
fixed-point iteration on an entropy-regularized energy with temperature 1/alpha, not plain GD on E.
Rules other than probs (hparams mcmc_update_rule, model/nlp/mcmc_update.py): gnorm and the line profile follow
the rule; Phi is recorded but only meaningful for probs; the Jacobian spectral radius is computed only for probs
(NaN otherwise).

Part A (convergence, 256 seqs): N = 32 steps at noise 0 (and the training noise if > 0); per token, per
  step: E_t, Phi_t, ||g_p||, cos(g_t, g_{t-1}), TV(p_{t+1}, p_t), log p(target) of softmax(z_t),
  entropy of p_t.
Part B (local geometry, 32 seqs, noise 0) at states t in {0, n_train - 1, N - 1}:
  - full Hessian of E w.r.t. the predicted embedding e = W_v p (D x D per token, block-diagonal over
    tokens): eigenvalue spectrum -> convexity and conditioning
  - spectral radius of the Jacobian of T (power iteration, Gelfand estimate): < 1 means locally
    contracting, the local linear convergence rate of the inference loop
  - line profile along the actual update: q(s) = softmax(p - s*alpha*g), s in [0, 2]: E, Phi, log p(target)
  - gradient geometry: share of ||g_p||^2 in the top 1% of vocab coordinates, kurtosis of g_p
  - effective dimension: participation ratio of the per-token embedding gradients g_e (<= D = 384)

Usage: python q3_energy_landscape/run_q3.py --model s2 --out-dir ...
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from harness import (RUN_DIRS, answer_masks, batches, energy, energy_from_embedding, entropy, eval_dataset,
                     eval_subset_indices, landscape_index, load_model, mcmc, token_logprob, train_eval_steps)
from model.nlp.mcmc_update import descent_direction, next_logits

LINE_S = np.linspace(0.0, 2.0, 17)


def phi(alpha, e, p):
    return alpha * e - entropy(p) - 0.5 * (p * p).sum(-1)


def part_a(model, val_ds, idx, n_steps, noise, alpha, batch_size, seed, rule="probs"):
    fields = ["energy", "phi", "gnorm", "cos_prev", "tv_next", "logp", "entropy"]
    rec = {k: [] for k in fields + ["logp_final", "answer"]}
    for b, ((inputs, targets, mask), amask) in enumerate(zip(batches(val_ds, idx, batch_size),
                                                             answer_masks(val_ds, idx, batch_size))):
        inputs, targets, mask = inputs.cuda(), targets.cuda(), mask.cuda()
        torch.manual_seed(seed + b)
        per = {k: [] for k in fields}
        gen = mcmc(model, inputs, n_steps, noise, alpha)
        prev_g, prev_p = None, None
        while True:
            try:
                s = next(gen)
            except StopIteration as stop:
                z_final = stop.value
                break
            g = descent_direction(rule, s.p, s.grad)  # the rule's step direction, softmax-null component removed
            per["energy"].append(s.energy[mask])
            per["phi"].append(phi(alpha, s.energy, s.p)[mask])
            per["gnorm"].append(g.norm(dim=-1)[mask])
            per["cos_prev"].append((F.cosine_similarity(g, prev_g, dim=-1) if prev_g is not None
                                    else torch.full_like(s.energy, float("nan")))[mask])
            per["logp"].append(token_logprob(s.z, targets)[mask])
            per["entropy"].append(entropy(s.p)[mask])
            if prev_p is not None:
                per["tv_next"].append((0.5 * (s.p - prev_p).abs().sum(-1))[mask])
            prev_g, prev_p = g, s.p
        p_final = F.softmax(z_final, -1)
        per["tv_next"].append((0.5 * (p_final - prev_p).abs().sum(-1))[mask])
        for k, v in per.items():
            rec[k].append(torch.stack(v).float().cpu())
        rec["logp_final"].append(token_logprob(z_final, targets)[mask].float().cpu())
        rec["answer"].append(amask.cuda()[mask].cpu())
        print(f"  A batch {b + 1}", flush=True)
    return {k: torch.cat(v, dim=-1).numpy() for k, v in rec.items()}


def capture_states(model, inputs, n_steps, alpha, wanted):
    states = {}
    gen = mcmc(model, inputs, n_steps, 0.0, alpha)
    for s in gen:
        if s.t in wanted:
            states[s.t] = s
    return states


def hessian_and_jacobian(model, ctx, s, alpha, mask, rule="probs", n_power=30):
    """Per-token Hessian spectrum of E(e) and spectral radius of the Jacobian of T at state s."""
    W = model.vocab_to_embed.weight  # (D, V)
    lsc = s.landscape
    e = model.vocab_to_embed(s.p).detach().requires_grad_()
    with torch.enable_grad():
        E = energy_from_embedding(model, ctx, e, lsc)
        (g_e,) = torch.autograd.grad(E.sum(), e, create_graph=True)
        g_e_tok = g_e.detach()[mask].cpu()
        hvp = lambda u: torch.autograd.grad(g_e, e, grad_outputs=u, retain_graph=True)[0].detach()
        D = e.shape[-1]
        H = torch.empty(*e.shape[:2], D, D, device=e.device)
        for k in range(D):
            u = torch.zeros_like(e)
            u[..., k] = 1.0
            H[..., :, k] = hvp(u)
        if rule == "probs":
            # Jacobian of T(p) = softmax(p - alpha * g_p(p)) applied to v: J_s(q) (v - alpha * W^T H_e W v)
            q = F.softmax(s.p - alpha * s.grad, -1)
            v = torch.randn_like(s.p)
            v = v / v.norm(dim=-1, keepdim=True)
            log_growth = torch.zeros(e.shape[:2], device=e.device)
            for _ in range(n_power):
                u = v - alpha * (hvp(v @ W.T) @ W)
                jv = q * u - q * (q * u).sum(-1, keepdim=True)
                n = jv.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                log_growth += n.squeeze(-1).log()
                v = jv / n
        else:  # T of the other rules is not derived here
            log_growth = torch.full(e.shape[:2], float("nan"), device=e.device)
    del g_e, E  # free the double-backward graph before the eigendecomposition
    H = H.detach()
    Hs = 0.5 * (H + H.transpose(-1, -2))[mask]
    del H
    eig = torch.linalg.eigvalsh(Hs.double()).float()  # (n_tok, D), ascending
    rho = (log_growth / n_power).exp().detach()[mask]
    g_p = (s.grad - s.grad.mean(-1, keepdim=True))[mask]
    g2 = g_p.pow(2)
    k = max(1, int(0.01 * g2.shape[-1]))
    top1pct_share = g2.topk(k, dim=-1).values.sum(-1) / g2.sum(-1)
    gz = (g_p - g_p.mean(-1, keepdim=True)) / g_p.std(-1, keepdim=True)
    kurtosis = gz.pow(4).mean(-1)
    return eig.cpu(), rho.cpu(), g_e_tok, top1pct_share.cpu(), kurtosis.cpu()


def line_profile(model, ctx, s, alpha, targets, mask, rule="probs"):
    nxt = landscape_index(model, s.t + 1)
    out = {"energy": [], "phi": [], "logp": []}
    with torch.no_grad():
        for sc in LINE_S:
            z = next_logits(rule, s.z, s.p, s.grad, sc * alpha)  # Part B runs at noise 0, so s.z is the logits the step starts from
            q = F.softmax(z, -1)
            e = energy(model, ctx, q, nxt)
            out["energy"].append(e[mask])
            out["phi"].append(phi(alpha, e, q)[mask])
            out["logp"].append(token_logprob(z, targets)[mask])
    return {k: torch.stack(v).cpu() for k, v in out.items()}


def participation_ratio(x: torch.Tensor) -> float:
    x = x / x.norm(dim=-1, keepdim=True)
    ev = torch.linalg.eigvalsh(torch.cov(x.double().T)).clamp_min(0)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def part_b(model, val_ds, idx, n_steps, n_train, alpha, batch_size, seed, rule="probs"):
    wanted = sorted({0, n_train - 1, n_steps - 1})
    res = {t: {k: [] for k in ["eig", "rho", "g_e", "top1pct", "kurt", "line_energy", "line_phi", "line_logp"]}
           for t in wanted}
    for b, (inputs, targets, mask) in enumerate(batches(val_ds, idx, batch_size)):
        inputs, targets, mask = inputs.cuda(), targets.cuda(), mask.cuda()
        torch.manual_seed(seed + b)
        states = capture_states(model, inputs, n_steps, alpha, wanted)
        ctx = model.embeddings(inputs).detach()
        for t, s in states.items():
            eig, rho, g_e, top, kurt = hessian_and_jacobian(model, ctx, s, alpha, mask, rule)
            line = line_profile(model, ctx, s, alpha, targets, mask, rule)
            for k, v in [("eig", eig), ("rho", rho), ("g_e", g_e), ("top1pct", top), ("kurt", kurt),
                         ("line_energy", line["energy"]), ("line_phi", line["phi"]), ("line_logp", line["logp"])]:
                res[t][k].append(v)
        del states, ctx
        torch.cuda.empty_cache()
        print(f"  B batch {b + 1} mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    out = {}
    for t, d in res.items():
        cat = {k: torch.cat(v, dim=0 if k in ("eig", "rho", "g_e", "top1pct", "kurt") else -1) for k, v in d.items()}
        out.update({f"t{t}_{k}": v.detach().numpy() for k, v in cat.items() if k != "g_e"})
        out[f"t{t}_g_e_participation_ratio"] = np.array(participation_ratio(cat["g_e"].detach()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(RUN_DIRS), required=True)
    ap.add_argument("--step", type=int, default=105000)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--n-seqs-a", type=int, default=256)
    ap.add_argument("--n-seqs-b", type=int, default=32)
    ap.add_argument("--batch-size-a", type=int, default=16)
    ap.add_argument("--batch-size-b", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--reuse-part-a", action="store_true", help="skip Part A if its arrays already exist")
    ap.add_argument("--data", default="rp_val", help="rp_val or an OOD set (ood_data.OOD_SETS)")
    ap.add_argument("--skip-part-b", action="store_true", help="Part A only (CE after k steps)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, hparams = load_model(args.model, args.step)
    val_ds = eval_dataset(args.data, hparams)
    idx = eval_subset_indices(len(val_ds), max(args.n_seqs_a, args.n_seqs_b))
    alpha = float(model.alpha.clamp(min=1e-4))
    rule = model.hparams.get("mcmc_update_rule", "probs")
    n_train = train_eval_steps(hparams)
    t0 = time.time()

    noises = sorted({0.0, float(hparams["langevin_dynamics_noise"])})
    for noise in noises:
        if args.reuse_part_a and (args.out_dir / f"part_a_noise{noise:g}.npz").exists():
            continue
        print(f"Part A noise={noise}", flush=True)
        arrays = part_a(model, val_ds, idx[: args.n_seqs_a], args.n_steps, noise, alpha, args.batch_size_a, args.seed, rule)
        np.savez_compressed(args.out_dir / f"part_a_noise{noise:g}.npz", **arrays)
    if not args.skip_part_b:
        print("Part B", flush=True)
        arrays = part_b(model, val_ds, idx[: args.n_seqs_b], args.n_steps, n_train, alpha, args.batch_size_b, args.seed, rule)
        np.savez_compressed(args.out_dir / "part_b.npz", **arrays, line_s=LINE_S)

    config = vars(args) | {"out_dir": str(args.out_dir), "alpha": alpha, "mcmc_update_rule": rule, "n_train_steps": n_train,
                           "noises": noises, "mcmc_num_steps": hparams["mcmc_num_steps"],
                           "seconds": round(time.time() - t0, 1)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
