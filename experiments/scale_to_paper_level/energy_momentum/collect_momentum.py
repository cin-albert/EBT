"""Pre-registered claims M1-M6 (IMPLEMENTATION_PLAN.md §0.4) -> report/result.md and report/figures/*.png.

Inputs: run_momentum.py outputs in results/single/<arm>_<set>/, run_bon.py outputs in results/bon/<arm>_k<k>_<set>/,
and T++ CE on the same 1,024 sequences from ../results/q2/s2_paper_step1000000_<set>_noise3/metrics.json.
Usage: python collect_momentum.py            (verdicts and tables; BoN rows only if results/bon exists)
       python collect_momentum.py --select   (print arm*, gd*, k* for run_bon.py, nothing else)
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent
sys.path.insert(0, str(WS.parent))
from collect_paper import pct_improvement  # noqa: E402

RES = WS / "results"
REPORT = WS / "report"
Q2 = WS.parent / "results/q2"
OOD_SETS = ["gsm8k", "squad", "bigbench_elementary_math_qa", "bigbench_dyck_languages"]
SETS = ["rp_val", *OOD_SETS]
SET_NAMES = {"rp_val": "Pretrain (val)", "gsm8k": "GSM8K", "squad": "SQuAD",
             "bigbench_elementary_math_qa": "BB Math QA", "bigbench_dyck_languages": "BB Dyck"}
N_STEPS = 32
K_LIST = (*range(1, 11), 16, 32)
K_EXTENDED = (5, 8, 10)
NOISES = (0.0, 3.0)
BON_N = (1, 2, 5, 10)
GD_TWO_CYCLE = 0.299  # ../report/report.md §2.5, rp_val, σ = 0, t = 31 (256 seqs)
PAPER_TABLE3 = {"tpp": [31.36, 49.6, 52.3, 79.8, 131.5], "ebt": [33.43, 43.3, 53.1, 72.6, 125.3]}  # in SETS order
OURS_FIG6A = {"tpp": 69.68, "k2": 91.68, "k3": 82.76, "bon10": 95.40}  # ../report/result.md


def ce_after_k(a: dict, k: int) -> np.ndarray:
    """Per-token CE after k updates: -logp[k] for k < N, -logp_final for k = N (Q3 Part A convention)."""
    n = a["logp"].shape[0]
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside 0..{n}")
    return -(a["logp_final"] if k == n else a["logp"][k])


def mean_ood_ppl(ce_by_set: dict[str, float], geometric: bool = False) -> float:
    """Arithmetic mean of per-set ppl (pre-registered); geometric = exp(mean CE over sets)."""
    ces = np.array(list(ce_by_set.values()))
    return float(np.exp(ces.mean())) if geometric else float(np.exp(ces).mean())


def paired_bootstrap(ce_a: np.ndarray, ce_b: np.ndarray, seq_id: np.ndarray, n: int = 1000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Token-mean of ce_a - ce_b and its 95% CI, resampling whole sequences."""
    _, inv = np.unique(seq_id, return_inverse=True)
    d_sum = np.bincount(inv, weights=ce_a - ce_b)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cnt), size=(n, len(cnt)))
    boot = d_sum[draws].sum(1) / cnt[draws].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d_sum.sum() / cnt.sum()), float(lo), float(hi)


def select(ce: dict[str, dict[str, dict[int, float]]]) -> tuple[str, str, int]:
    """(arm*, gd*, k*) from rp_val only: lowest CE at k = 3 among momentum / GD arms; k* in K_EXTENDED for arm*."""
    rp = {arm: v["rp_val"] for arm, v in ce.items()}
    arm_star = min((a for a in rp if not a.startswith("gd_")), key=lambda a: rp[a][3])
    gd_star = min((a for a in rp if a.startswith("gd_")), key=lambda a: rp[a][3])
    k_star = min(K_EXTENDED, key=lambda k: rp[arm_star][k])
    return arm_star, gd_star, k_star


def verdict_m1(pct_mean_ood: float, ci_hi_by_set: list[float]) -> bool:
    return pct_mean_ood > 2.0 and sum(hi < 0 for hi in ci_hi_by_set) >= 3


def phase3_gate(supported: dict[str, bool]) -> bool:
    return not supported["M1"] and not supported["M3"]


def bon_ce(e: np.ndarray, logp: np.ndarray, n: int) -> float:
    """Mean CE when each token keeps the lowest-energy one of its first n candidates. e, logp: (G, tokens)."""
    pick = e[:n].argmin(0)
    return float(-logp[pick, np.arange(logp.shape[1])].mean())


# ---------------------------------------------------------------- loading and tables
C_EBT, C_TPP, C_ARM, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"  # categorical slots 1-3; gray for the control
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"  # diverging blue <-> red, gray midpoint


def load_single(arm: str, data: str, noise: float) -> dict:
    return dict(np.load(RES / "single" / f"{arm}_{data}" / f"part_a_noise{noise:g}.npz"))


def arms_done() -> list[str]:
    from momentum import ARMS
    return [a for a in ARMS if all((RES / "single" / f"{a}_{s}" / "config.json").exists() for s in SETS)]


def summarize(arms: list[str]) -> dict:
    """ce[arm][set][noise][k] = mean CE over tokens."""
    return {arm: {s: {nz: {k: float(ce_after_k(a, k).mean()) for k in K_LIST}
                      for nz in NOISES for a in [load_single(arm, s, nz)]} for s in SETS} for arm in arms}


def tpp_ce() -> dict[str, float]:
    return {s: json.loads((Q2 / f"s2_paper_step1000000_{s}_noise3/metrics.json").read_text())["single_sample"]["tpp_ce"]
            for s in SETS}


def pct_ppl(base: float, variant: float) -> float:
    """% ppl improvement of variant over base (ppl values)."""
    return 100 * (1 - variant / base)


def ood_ppl(ce: dict, arm: str, noise: float, k: int, geometric: bool = False) -> float:
    return mean_ood_ppl({s: ce[arm][s][noise][k] for s in OOD_SETS}, geometric)


def bon_results(arm: str, k: int) -> dict[str, dict[int, float]] | None:
    """CE of BoN-N per set, or None if any set is missing."""
    out = {}
    for s in SETS:
        f = RES / "bon" / f"{arm}_k{k}_{s}" / "arrays.npz"
        if not f.exists():
            return None
        a = np.load(f)
        out[s] = {n: bon_ce(a["e_final"], a["logp"], n) for n in BON_N}
    return out


def two_cycle_share(arm: str) -> float:
    from analyze_q3_extra import cycle_metrics
    return cycle_metrics(load_single(arm, "rp_val", 0.0))["share_two_cycle"]


# ---------------------------------------------------------------- claims
def claims(ce: dict, tpp: dict, arm_star: str, gd_star: str, k_star: int) -> tuple[list[dict], dict[str, bool]]:
    rows, ok = [], {}
    base = ood_ppl(ce, "gd_a5", 3.0, 3)
    star3 = ood_ppl(ce, arm_star, 3.0, 3)

    his = []
    for s in OOD_SETS:
        a, g = load_single(arm_star, s, 3.0), load_single("gd_a5", s, 3.0)
        his.append(paired_bootstrap(ce_after_k(a, 3), ce_after_k(g, 3), a["seq_id"])[2])
    p1 = pct_ppl(base, star3)
    ok["M1"] = verdict_m1(p1, his)
    rows.append({"id": "M1", "value": f"{base:.2f} → {star3:.2f} ({p1:+.1f}%); CI < 0 on {sum(h < 0 for h in his)}/4",
                 "rule": "> 2% and CI < 0 on ≥ 3/4"})

    gd3 = ood_ppl(ce, gd_star, 3.0, 3)
    p2 = pct_ppl(gd3, star3)
    ok["M2"] = p2 > 1.0
    rows.append({"id": "M2", "value": f"{gd_star} {gd3:.2f} vs {arm_star} {star3:.2f} ({p2:+.1f}%)", "rule": "> 1%"})

    stark = ood_ppl(ce, arm_star, 3.0, k_star)
    p3 = pct_ppl(star3, stark)
    ok["M3"] = p3 > 2.0
    rows.append({"id": "M3", "value": f"k = 3 {star3:.2f} → k = {k_star} {stark:.2f} ({p3:+.1f}%)", "rule": "> 2%"})

    cyc = two_cycle_share(arm_star)
    ok["M4"] = cyc < 0.15
    rows.append({"id": "M4", "value": f"{100 * cyc:.1f}% (gd_a5 here: {100 * two_cycle_share('gd_a5'):.1f}%)",
                 "rule": "< 15%"})

    bon = bon_results(arm_star, 3)
    if bon is None:
        ok["M5"] = None
        rows.append({"id": "M5", "value": "pending (no BoN outputs)", "rule": "> 0"})
    else:
        one = mean_ood_ppl({s: bon[s][1] for s in OOD_SETS})
        ten = mean_ood_ppl({s: bon[s][10] for s in OOD_SETS})
        p5 = pct_ppl(one, ten)
        ok["M5"] = p5 > 0
        rows.append({"id": "M5", "value": f"1 sample {one:.2f} → BoN-10 {ten:.2f} ({p5:+.1f}%)", "rule": "> 0"})

    wins = [s for s in OOD_SETS if ce[arm_star][s][3.0][3] < tpp[s]]
    ok["M6"] = len(wins) >= 1
    rows.append({"id": "M6", "value": f"{len(wins)} of 4" + (f" ({', '.join(SET_NAMES[s] for s in wins)})" if wins else ""),
                 "rule": "≥ 1 of 4"})
    for r in rows:
        r["verdict"] = {True: "**Supported**", False: "**Not supported**", None: "pending"}[ok[r["id"]]]
    return rows, ok


# ---------------------------------------------------------------- figures
def plot_ce_vs_k(ce: dict, tpp: dict, arm_star: str, gd_star: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tpp_mean = mean_ood_ppl({s: tpp[s] for s in OOD_SETS})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, noise in zip(axes, NOISES):
        lines = [("gd_a5 (GD, trained α)", "gd_a5", C_EBT, "-"), (f"{gd_star} (GD control)", gd_star, MUTED, "--"),
                 (f"{arm_star} (arm*)", arm_star, C_ARM, "-")]
        for label, arm, color, ls in dict.fromkeys(lines):
            y = [ood_ppl(ce, arm, noise, k) for k in K_LIST]
            ax.plot(K_LIST, y, ls, color=color, lw=2, marker="o", ms=4, label=label)
        ax.axhline(tpp_mean, color=C_TPP, ls=":", lw=2, label="Transformer++")
        ax.axvline(3, color=MUTED, lw=1, alpha=0.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_LIST, [str(k) for k in K_LIST], fontsize=8)
        ax.set_title(f"σ = {noise:g}", fontsize=10, color="#3d3d3a")
        ax.set_xlabel("MCMC steps k (trained: 3)")
        ax.grid(alpha=0.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean OOD perplexity (4 sets)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grid_heatmap(ce: dict, arms: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    d = np.array([[ce[a]["rp_val"][nz][3] - ce["gd_a5"]["rp_val"][nz][3] for nz in NOISES] for a in arms])
    cmap = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
    lim = 0.3
    fig, ax = plt.subplots(figsize=(4.6, 0.36 * len(arms) + 1.2))
    ax.imshow(np.clip(d, -lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d[i, j]:+.3f}", ha="center", va="center", fontsize=8, color="#1f1f1d")
    ax.set_xticks(range(len(NOISES)), [f"σ = {nz:g}" for nz in NOISES])
    ax.set_yticks(range(len(arms)), arms, fontsize=8)
    ax.set_title("ΔCE vs gd_a5, rp_val, k = 3 (blue = better; color clipped at ±0.3)", fontsize=8, color="#3d3d3a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bon(bons: dict[str, dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gd_a5": C_EBT}
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for arm, bon in bons.items():
        y = [mean_ood_ppl({s: bon[s][n] for s in OOD_SETS}) for n in BON_N]
        ax.plot(BON_N, y, "-", color=colors.get(arm, C_ARM), lw=2, marker="o", ms=5, label=arm)
    ax.set_xticks(BON_N)
    ax.set_xlabel("candidates N (lowest final energy kept), k = 3, σ = 3")
    ax.set_ylabel("mean OOD perplexity")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- report
def write_result(ce: dict, tpp: dict, arms: list[str], sel: tuple[str, str, int], rows: list[dict],
                 ok: dict, bons: dict) -> str:
    arm_star, gd_star, k_star = sel
    L = ["# Energy momentum results (generated by collect_momentum.py)", "",
         f"Selected on rp_val at k = 3, σ = 3: **arm\\* = {arm_star}**, **gd\\* = {gd_star}**, **k\\* = {k_star}**.", "",
         "## Pre-registered claims", "", "| # | Value | Supported if | Verdict |", "|---|---|---|---|"]
    L += [f"| {r['id']} | {r['value']} | {r['rule']} | {r['verdict']} |" for r in rows]
    gate = "pending" if any(v is None for v in ok.values()) else phase3_gate(ok)
    L += ["", f"**phase3_gate = {gate}** (M1 and M3 both not supported)", ""]

    L += ["## Sanity check: gd_a5 against ../report/result.md (mean OOD ppl, σ = 3)", "",
          "| k | gd_a5 here | report |", "|---|---|---|",
          f"| 2 | {ood_ppl(ce, 'gd_a5', 3.0, 2):.2f} | {OURS_FIG6A['k2']} |",
          f"| 3 | {ood_ppl(ce, 'gd_a5', 3.0, 3):.2f} | {OURS_FIG6A['k3']} |", ""]

    L += ["## Per-set perplexity at k = 3 (σ = 3), with baselines", "",
          "| Set | T++ paper | EBT paper | T++ ours | gd_a5 | " + f"{gd_star} | {arm_star} |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(SETS):
        L.append(f"| {SET_NAMES[s]} | {PAPER_TABLE3['tpp'][i]} | {PAPER_TABLE3['ebt'][i]} | {math.exp(tpp[s]):.2f} | "
                 f"{math.exp(ce['gd_a5'][s][3.0][3]):.2f} | {math.exp(ce[gd_star][s][3.0][3]):.2f} | "
                 f"{math.exp(ce[arm_star][s][3.0][3]):.2f} |")
    L.append(f"| Mean of 4 OOD | {np.mean(PAPER_TABLE3['tpp'][1:]):.1f} | {np.mean(PAPER_TABLE3['ebt'][1:]):.1f} | "
             f"{mean_ood_ppl({s: tpp[s] for s in OOD_SETS}):.2f} | {ood_ppl(ce, 'gd_a5', 3.0, 3):.2f} | "
             f"{ood_ppl(ce, gd_star, 3.0, 3):.2f} | {ood_ppl(ce, arm_star, 3.0, 3):.2f} |")

    for nz in NOISES:
        L += ["", f"## Mean OOD ppl vs k (σ = {nz:g}; geometric mean in the last column at k = 3)", "",
              "| Arm | " + " | ".join(f"k={k}" for k in K_LIST) + " | geo k=3 |", "|---" * (len(K_LIST) + 2) + "|"]
        for a in arms:
            L.append(f"| {a} | " + " | ".join(f"{ood_ppl(ce, a, nz, k):.1f}" for k in K_LIST)
                     + f" | {ood_ppl(ce, a, nz, 3, geometric=True):.1f} |")
        L += ["", f"## rp_val CE vs k (σ = {nz:g})", "",
              "| Arm | " + " | ".join(f"k={k}" for k in K_LIST) + " |", "|---" * (len(K_LIST) + 1) + "|"]
        for a in arms:
            L.append(f"| {a} | " + " | ".join(f"{ce[a]['rp_val'][nz][k]:.4f}" for k in K_LIST) + " |")

    L += ["", "## 2-cycles (rp_val, σ = 0, t = 31)", "", "| Arm | share two-cycle |", "|---|---|"]
    L += [f"| {a} | {100 * two_cycle_share(a):.1f}% |" for a in arms]

    if bons:
        L += ["", "## Self-verification (k = 3, σ = 3): mean OOD ppl and per-set % improvement of BoN-10 over 1 sample", "",
              "| Arm | " + " | ".join(f"BoN-{n}" for n in BON_N) + " | " + " | ".join(SET_NAMES[s] for s in SETS) + " |",
              "|---" * (len(BON_N) + len(SETS) + 1) + "|"]
        for arm, bon in bons.items():
            L.append(f"| {arm} | " + " | ".join(f"{mean_ood_ppl({s: bon[s][n] for s in OOD_SETS}):.2f}" for n in BON_N)
                     + " | " + " | ".join(f"{pct_improvement(bon[s][1], bon[s][10]):+.1f}%" for s in SETS) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true", help="print 'arm* gd* k*' and exit")
    args = ap.parse_args()
    arms = arms_done()
    ce = summarize(arms)
    sel = select({a: {s: ce[a][s][3.0] for s in SETS} for a in arms})
    if args.select:
        print(*sel)
        return
    tpp = tpp_ce()
    rows, ok = claims(ce, tpp, *sel)
    bons = {a: b for a in dict.fromkeys(["gd_a5", sel[0]]) if (b := bon_results(a, 3)) is not None}
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    (REPORT / "result.md").write_text(write_result(ce, tpp, arms, sel, rows, ok, bons))
    plot_ce_vs_k(ce, tpp, sel[0], sel[1], REPORT / "figures/ce_vs_k.png")
    plot_grid_heatmap(ce, arms, REPORT / "figures/grid_heatmap.png")
    if bons:
        plot_bon(bons, REPORT / "figures/bon.png")
    print((REPORT / "result.md").read_text())


if __name__ == "__main__":
    main()
