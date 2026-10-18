"""Extra Q3 metrics (gradient magnitude, curvature shape, sharpness, line smoothness, 2-cycles) from run_q3.py arrays.

Offline only: reads the saved arrays, no model forward passes.

Usage: python q3_energy_landscape/analyze_q3_extra.py RUN_DIR [RUN_DIR ...]   -> RUN_DIR/metrics_extra.json
"""
import json
import sys
from pathlib import Path

import numpy as np

TV_CONVERGED = 0.01  # same threshold as analyze_q3.py
STRONG = 0.1  # an eigenvalue is "strong" if |lambda| > STRONG * lambda_max of that token
FLAT = 0.01  # an eigenvalue is "flat" if |lambda| < FLAT * max|lambda| of that token
CYCLE_RATIO = 0.1  # 2-cycle: |logp_31 - logp_29| < CYCLE_RATIO * |logp_31 - logp_30|


def quantiles(x) -> dict:
    return {f"p{p}": float(np.nanpercentile(x, p)) for p in (1, 50, 99)}


def gradient_metrics(a, n_train: int, alpha: float) -> dict:
    g = a["gnorm"]  # (steps, tokens), ||g_p|| with the softmax-null component removed
    last, tr = g.shape[0] - 1, n_train - 1  # tr = last trained state, as in Part B
    ratio = g[last] / g[tr]
    ce = -a["logp_final"]
    return {
        "gnorm_per_step": {t: quantiles(g[t]) for t in sorted({0, 1, 2, 3, 5, 10, last})},
        "alpha_x_median_gnorm_trained": float(alpha * np.median(g[tr])),
        "ratio_last_over_trained": quantiles(ratio),
        "share_ratio_below_0.1": float((ratio < 0.1).mean()),
        "share_ratio_above_10": float((ratio > 10).mean()),
        "corr_log_gnorm_trained_vs_final_ce": float(np.corrcoef(np.log(g[tr] + 1e-12), ce)[0, 1]),
    }


def cycle_metrics(a) -> dict:
    """Tokens not converged at the last step, and the share of them that flip-flop with period 2."""
    tv, lp = a["tv_next"], a["logp"]
    not_conv = tv[-2] >= TV_CONVERGED
    d1, d2 = np.abs(lp[-1] - lp[-2]), np.abs(lp[-1] - lp[-3])
    cyc = not_conv & (d2 < CYCLE_RATIO * d1)
    ce_after = -np.concatenate([lp[1:], a["logp_final"][None]], 0).mean(1)
    dE = np.abs(np.diff(a["energy"][10:], axis=0))
    return {
        "share_not_converged": float(not_conv.mean()),
        "share_two_cycle": float(cyc.mean()),
        "two_cycle_share_of_not_converged": float(cyc.sum() / max(not_conv.sum(), 1)),
        "median_tv_two_cycle": float(np.median(tv[-2][cyc])) if cyc.any() else None,
        "ce_last_two_steps": [float(ce_after[-2]), float(ce_after[-1])],
        "mean_abs_dE_after_step10": float(dE.mean()),
        "share_energy_non_monotone": float((np.diff(a["energy"][1:], axis=0) > 1e-6).any(0).mean()),
    }


def hessian_metrics(ev, rho, alpha: float) -> dict:
    lmax, lmin = ev[:, -1], ev[:, 0]
    absev = np.abs(ev)
    trace = ev.sum(1)
    return {
        "lambda_max": quantiles(lmax),
        "lambda_min": quantiles(lmin),
        "alpha_x_median_lambda_max": float(alpha * np.median(lmax)),
        "share_neg_eig": quantiles((ev < 0).mean(1)),
        "share_flat_eig": quantiles((absev < FLAT * absev.max(1, keepdims=True)).mean(1)),
        "n_strong_pos": quantiles((ev > STRONG * lmax[:, None]).sum(1)),
        "n_strong_neg": quantiles((ev < -STRONG * lmax[:, None]).sum(1)),
        "share_trace_negative": float((trace < 0).mean()),
        "share_abs_lmin_gt_lmax": float((-lmin > lmax).mean()),
        "rho": quantiles(rho),
        "share_rho_above_1": float((rho > 1).mean()),
    }


def line_metrics(L) -> dict:
    """L: (n_s, tokens) profile along z = p - s * alpha * grad."""
    d1, d2 = np.diff(L, axis=0), np.diff(L, 2, axis=0)
    roughness = np.abs(d2).max(0) / (np.abs(L[-1] - L[0]) + 1e-8)
    return {
        "slope_sign_changes": quantiles((np.diff(np.sign(d1), axis=0) != 0).sum(0)),
        "share_convex": float((d2 >= -1e-6).all(0).mean()),
        "roughness": quantiles(roughness),
    }


def main():
    for run_dir in map(Path, sys.argv[1:]):
        cfg = json.loads((run_dir / "metrics.json").read_text())["config"]
        alpha, n_train = cfg["alpha"], cfg["n_train_steps"]
        out = {"config": {"alpha": alpha, "n_train_steps": n_train, "strong": STRONG, "flat": FLAT,
                          "cycle_ratio": CYCLE_RATIO, "tv_threshold": TV_CONVERGED}}
        for f in sorted(run_dir.glob("part_a_noise*.npz")):
            a = np.load(f)
            out[f.stem] = {"gradient": gradient_metrics(a, n_train, alpha), "cycles": cycle_metrics(a)}
        b = np.load(run_dir / "part_b.npz")
        states = sorted({int(k[1:].split("_")[0]) for k in b.files if k.startswith("t")})
        out["part_b"] = {f"t{t}": {"hessian": hessian_metrics(b[f"t{t}_eig"], b[f"t{t}_rho"], alpha),
                                   "line": {nm: line_metrics(b[f"t{t}_line_{nm}"]) for nm in ("energy", "phi", "logp")}}
                         for t in states}
        (run_dir / "metrics_extra.json").write_text(json.dumps(out, indent=2))
        print(f"wrote {run_dir / 'metrics_extra.json'}")


if __name__ == "__main__":
    main()
