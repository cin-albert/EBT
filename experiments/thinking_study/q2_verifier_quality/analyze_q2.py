"""Q2 metrics from run_q2.py arrays: verifier consistency, best-of-N efficiency, uncertainty, off-trajectory.

Usage: python q2_verifier_quality/analyze_q2.py RUN_DIR [RUN_DIR ...]   -> RUN_DIR/metrics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

BON_SIZES = (1, 2, 4, 8, 16, 32)
MIN_CE_GAP = 0.1  # nats; pairs closer than this are "ties" for the meaningful-difference consistency


def auroc(score, positive):
    """P(score of a positive > score of a negative), ties counted half (Mann-Whitney)."""
    r = rankdata(score)
    n_pos = positive.sum()
    n_neg = len(positive) - n_pos
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pairwise_consistency(e, ce, min_gap, chunk=20000):
    """Share of same-token candidate pairs with |dCE| > min_gap where lower energy has lower CE."""
    G = e.shape[0]
    iu, ju = np.triu_indices(G, 1)
    agree = total = 0
    for s in range(0, e.shape[1], chunk):
        de = e[iu, s:s + chunk] - e[ju, s:s + chunk]
        dc = ce[iu, s:s + chunk] - ce[ju, s:s + chunk]
        valid = (np.abs(dc) > min_gap) & (de != 0)
        agree += ((np.sign(de) == np.sign(dc)) & valid).sum()
        total += valid.sum()
    return float(agree / total) if total else float("nan"), int(total)


def best_of_n(e, ce):
    out = {}
    for n in BON_SIZES:
        if n > e.shape[0]:
            break
        en, cn = e[:n], ce[:n]
        sel = np.take_along_axis(cn, en.argmin(0)[None], 0)[0].mean()
        worst = np.take_along_axis(cn, en.argmax(0)[None], 0)[0].mean()
        oracle = cn.min(0).mean()
        rand = cn.mean()
        eff = (rand - sel) / (rand - oracle) if rand > oracle else float("nan")
        out[n] = {"ce_energy_pick": float(sel), "ce_random": float(rand), "ce_oracle": float(oracle),
                  "ce_max_energy_pick": float(worst), "verifier_efficiency": float(eff)}
    return out


def analyze(run_dir: Path) -> dict:
    a = np.load(run_dir / "arrays.npz")
    cfg = json.loads((run_dir / "config.json").read_text())
    ce = -a["logp"]
    m = {"config": cfg}
    m["single_sample"] = {"ce": float(ce[0].mean()), "ppl": float(np.exp(ce[0].mean())),
                          "ce_all_candidates": float(ce.mean()), "tpp_ce": float(-a["tpp_logp"].mean())}
    m["diversity"] = {"mean_js_cand0_vs_1to3": float(a["js"].mean()),
                      "mean_std_ce_across_candidates": float(ce.std(0).mean()),
                      "share_tokens_ce_range_below_1e-3": float(((ce.max(0) - ce.min(0)) < 1e-3).mean())}
    for key in ("e_final", "e_last_logged"):
        e = a[key]
        m[key] = {
            "pairwise_consistency_any_gap": pairwise_consistency(e, ce, 1e-6),
            f"pairwise_consistency_gap_gt_{MIN_CE_GAP}": pairwise_consistency(e, ce, MIN_CE_GAP),
            "best_of_n": best_of_n(e, ce),
            # tests whether energy prefers sharper candidates: share of pairs where lower E has lower entropy
            "pairwise_lower_energy_has_lower_entropy": pairwise_consistency(e, a["entropy"], 1e-3),
        }
    wrong = ~a["top1"][0].astype(bool)
    m["uncertainty"] = {
        "auroc_energy_predicts_top1_error": auroc(a["e_final"][0], wrong),
        "auroc_entropy_predicts_top1_error": auroc(a["entropy"][0], wrong),
        "spearman_energy_vs_ce_across_tokens": float(spearmanr(a["e_final"][0], ce[0]).statistic),
        "spearman_entropy_vs_ce_across_tokens": float(spearmanr(a["entropy"][0], ce[0]).statistic),
        "top1_error_rate": float(wrong.mean()),
    }
    names = ["ebt_cand0"] + cfg["off_trajectory"]
    e7 = np.concatenate([a["e_final"][:1], a["off_e"]], 0)
    ce7 = np.concatenate([ce[:1], -a["off_logp"]], 0)
    lowest = e7.argmin(0)
    m["off_trajectory"] = {
        "mean_energy": {n: float(e7[i].mean()) for i, n in enumerate(names)},
        "mean_ce": {n: float(ce7[i].mean()) for i, n in enumerate(names)},
        "share_lowest_energy": {n: float((lowest == i).mean()) for i, n in enumerate(names)},
        "share_onehot_target_below_ebt_pred": float((e7[names.index("onehot_target")] < e7[0]).mean()),
        "share_tpp_T1_below_ebt_pred": float((e7[names.index("tpp_T1.0")] < e7[0]).mean()),
        "pairwise_consistency_gap_gt_0.1": pairwise_consistency(e7, ce7, MIN_CE_GAP),
        "bon_over_7_candidates_ce": float(np.take_along_axis(ce7, lowest[None], 0)[0].mean()),
    }
    return m


def main():
    for d in map(Path, sys.argv[1:]):
        m = analyze(d)
        (d / "metrics.json").write_text(json.dumps(m, indent=1))
        print(d.name, json.dumps({k: v for k, v in m.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
