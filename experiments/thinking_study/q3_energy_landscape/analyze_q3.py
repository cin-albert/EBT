"""Q3 metrics from run_q3.py arrays.

Usage: python q3_energy_landscape/analyze_q3.py RUN_DIR [RUN_DIR ...]   -> RUN_DIR/metrics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

TV_CONVERGED = 0.01  # total-variation change between consecutive predictions counted as "converged"


def first_below(x, thr):
    """Index of the first step with x < thr per token (len(x) if never)."""
    below = x < thr
    return np.where(below.any(0), below.argmax(0), x.shape[0])


def part_a_metrics(a, n_train):
    ce = -a["logp"]  # ce[t] = CE of softmax(z_t); z_0 is the initial noise
    ce_after = np.concatenate([ce[1:], -a["logp_final"][None]], 0)  # ce_after[k-1] = CE after k steps
    n = ce_after.shape[0]
    m = {"ce_after_k_steps": {k: float(ce_after[k - 1].mean()) for k in range(1, n + 1)},
         "mean_per_step": {k: {t: float(np.nanmean(a[k][t])) for t in range(a[k].shape[0])}
                           for k in ("energy", "phi", "entropy", "gnorm", "tv_next", "cos_prev")}}
    best_k = int(np.argmin(ce_after.mean(1))) + 1
    m["best_k"] = {"k": best_k, "ce": float(ce_after[best_k - 1].mean()),
                   "ce_at_train_steps": float(ce_after[n_train - 1].mean())}
    conv = first_below(a["tv_next"], TV_CONVERGED)
    m["convergence"] = {"tv_threshold": TV_CONVERGED, "median_steps": float(np.median(conv)),
                        "share_converged_by_train_steps": float((conv < n_train).mean()),
                        "share_converged_within_run": float((conv < a["tv_next"].shape[0]).mean()),
                        "steps_quantiles": {q: float(np.quantile(conv, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
    # energy computed at the state the step starts from: dE[t] = E[t+1] - E[t]; loss of that state = ce[t]
    dE = np.diff(a["energy"], axis=0)
    dCE = np.diff(ce, axis=0)
    dPhi = np.diff(a["phi"], axis=0)
    m["alignment"] = {
        "share_E_down_CE_up": float(((dE < 0) & (dCE > 0)).mean()),
        "share_E_down": float((dE < 0).mean()),
        "share_phi_down": float((dPhi < 0).mean()),
        "share_E_down_CE_up_per_step": {t: float(((dE[t] < 0) & (dCE[t] > 0)).mean()) for t in range(dE.shape[0])},
        "spearman_dE_dCE_per_step": {t: float(spearmanr(dE[t], dCE[t]).statistic) for t in range(min(8, dE.shape[0]))},
    }
    ce_train = ce_after[n_train - 1]
    m["adaptivity"] = {
        "spearman_steps_to_converge_vs_ce": float(spearmanr(conv, ce_train).statistic),
        "spearman_steps_to_converge_vs_entropy": float(spearmanr(conv, a["entropy"][n_train - 1]).statistic),
        "spearman_energy_vs_ce_at_train_steps": float(spearmanr(a["energy"][n_train - 1], ce_train).statistic),
    }
    return m


def part_b_metrics(b):
    m = {}
    for t in sorted({int(k.split("_")[0][1:]) for k in b.files if k.startswith("t")}):
        eig = b[f"t{t}_eig"]
        lam_max = eig[:, -1]
        abs_med = np.median(np.abs(eig), axis=1)
        line_ce = -b[f"t{t}_line_logp"]
        s = b["line_s"]
        m[t] = {
            "hessian_E_wrt_embedding": {
                "median_lambda_min": float(np.median(eig[:, 0])), "median_lambda_max": float(np.median(lam_max)),
                "share_negative_eigenvalues": float((eig < 0).mean()),
                "share_tokens_locally_convex": float((eig[:, 0] >= -1e-3 * np.abs(lam_max)).mean()),
                "median_ratio_lambda_max_to_median_abs": float(np.median(lam_max / abs_med)),
                "median_spectrum_quantiles": {q: float(np.median(np.quantile(eig, q, axis=1))) for q in (0.0, 0.1, 0.5, 0.9, 1.0)},
            },
            "update_map_jacobian": {"median_spectral_radius": float(np.median(b[f"t{t}_rho"])),
                                    "share_contracting": float((b[f"t{t}_rho"] < 1).mean()),
                                    "quantiles": {q: float(np.quantile(b[f"t{t}_rho"], q)) for q in (0.1, 0.5, 0.9)}},
            "gradient_geometry": {"median_top1pct_vocab_share_of_g2": float(np.median(b[f"t{t}_top1pct"])),
                                  "median_kurtosis": float(np.median(b[f"t{t}_kurt"])),
                                  "embedding_gradient_participation_ratio": float(b[f"t{t}_g_e_participation_ratio"])},
            "line_along_update": {
                "s": s.tolist(),
                "mean_ce": line_ce.mean(1).tolist(),
                "mean_energy": b[f"t{t}_line_energy"].mean(1).tolist(),
                "mean_phi": b[f"t{t}_line_phi"].mean(1).tolist(),
                "median_argmin_s_ce": float(np.median(s[line_ce.argmin(0)])),
                "median_argmin_s_energy": float(np.median(s[b[f"t{t}_line_energy"].argmin(0)])),
                "median_argmin_s_phi": float(np.median(s[b[f"t{t}_line_phi"].argmin(0)])),
            },
        }
    return m


def analyze(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config.json").read_text())
    m = {"config": cfg}
    for noise in cfg["noises"]:
        m[f"part_a_noise{noise:g}"] = part_a_metrics(np.load(run_dir / f"part_a_noise{noise:g}.npz"), cfg["n_train_steps"])
    m["part_b"] = part_b_metrics(np.load(run_dir / "part_b.npz"))
    return m


def main():
    for d in map(Path, sys.argv[1:]):
        m = analyze(d)
        (d / "metrics.json").write_text(json.dumps(m, indent=1))
        print(d.name, "written")


if __name__ == "__main__":
    main()
