"""Pre-registered claims C1-C7 (IMPLEMENTATION_PLAN.md §0.2) -> report/result.md and report/figures/*.png.

Inputs: validation CSVs of both runs (runs/logs/csv), and run_eval.sh outputs in results/{q2,q3}.
Usage: python collect_paper.py
"""
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent
CSV = WS / "runs/logs/csv"
RES = WS / "results"
REPORT = WS / "report"
OOD_SETS = ["gsm8k", "squad", "bigbench_elementary_math_qa", "bigbench_dyck_languages"]
SETS = ["rp_val", *OOD_SETS]
TOKENS_PER_STEP = 128 * 256  # global batch x context
BON_N = (2, 5, 10)
STEPS_PER_CANDIDATE = 3
C_TPP, C_EBT, MUTED = "#eb6834", "#2a78d6", "#8a8984"
SET_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5, in SETS order
SET_NAMES = {"rp_val": "Pretrain (val)", "gsm8k": "GSM8K", "squad": "SQuAD",
             "bigbench_elementary_math_qa": "BB Math QA", "bigbench_dyck_languages": "BB Dyck"}
PAPER_TABLE3 = {"tpp": [31.36, 49.6, 52.3, 79.8, 131.5], "ebt": [33.43, 43.3, 53.1, 72.6, 125.3]}  # in SETS order
PAPER = {"C1": "+0.064", "C2": "≈ 20%", "C3": "≈ 11%", "C4": "4–8% → 10–14%", "C5": "qualitative",
         "C6": "linear, 12% → 23%", "C7": "3 of 4"}


def pct_improvement(ce_base: float, ce_variant: float) -> float:
    """Paper's % perplexity improvement: 100 (1 - ppl_variant / ppl_base)."""
    return 100 * (1 - math.exp(ce_variant - ce_base))


def bon_ce(e: np.ndarray, ce: np.ndarray, n: int) -> float:
    """Mean CE of the lowest-energy candidate among the first n; e, ce are (candidates, tokens)."""
    pick = e[:n].argmin(0)
    return float(np.take_along_axis(ce[:n], pick[None], 0).mean())


def spearman(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def val_curve(run_csv_dir: Path, col: str) -> dict[int, float]:
    """{checkpoint step: validation loss}; the first row per step wins, because a resumed run repeats the
    checkpoint's validation on its 1 remaining batch (ledger ruling, Task 6)."""
    out = {}
    for path in sorted(Path(run_csv_dir).glob("version_*/metrics.csv"), key=lambda p: int(p.parent.name[8:])):
        for r in csv.DictReader(open(path)):
            if r.get(col):
                out.setdefault(int(r["step"]) + 1, float(r[col]))  # CSV step is 0-based; checkpoint names count steps
    return dict(sorted(out.items()))


def best_step(curve: dict[int, float]) -> int:
    return min(curve, key=curve.get)


def mean_ppl(ces) -> float:
    return float(np.mean(np.exp(ces)))


def verdicts(results: dict) -> list[dict]:
    """One row per claim: value, threshold, verdict ("Supported", "Not supported", or C1's "(matches paper)")."""
    yes = lambda ok: "Supported" if ok else "Not supported"
    rows = []
    gap = results["gap"]
    c1 = "Supported (matches paper)" if gap <= 0.114 else yes(gap < 0.149)
    rows.append({"claim": "C1", "value": f"gap {gap:+.3f}", "threshold": "< 0.149 (paper match ≤ 0.114)", "verdict": c1})
    m = results["mean_ppl"]
    c2 = 100 * (1 - m["k3"] / m["k2"])
    rows.append({"claim": "C2", "value": f"{m['k2']:.2f} → {m['k3']:.2f} ({c2:.1f}%)", "threshold": "> 10%",
                 "verdict": yes(c2 > 10)})
    c3 = 100 * (1 - m["bon10"] / m["k3"])
    rows.append({"claim": "C3", "value": f"{m['k3']:.2f} → {m['bon10']:.2f} ({c3:.1f}%)", "threshold": "> 2%",
                 "verdict": yes(c3 > 2)})
    for claim, key, what in [("C4", "dyck_curve", "Dyck BoN-5 vs 1 sample"), ("C5", "val_curve", "val BoN-10 vs BoN-2")]:
        curve = results[key]
        final, rho = curve[max(curve)], spearman(list(curve), list(curve.values()))
        rows.append({"claim": claim, "value": f"{what}: final {final:+.2f}%, ρ = {rho:+.2f}",
                     "threshold": "final > 0 and ρ > 0", "verdict": yes(final > 0 and rho > 0)})
    x, y = zip(*results["shift_points"])
    rho = spearman(x, y)
    rows.append({"claim": "C6", "value": f"ρ(shift, Max Thinking %) = {rho:+.2f}", "threshold": "> 0",
                 "verdict": yes(rho > 0)})
    wins = sum(s2 < tpp for s2, tpp in results["ood_ppl"].values())
    rows.append({"claim": "C7", "value": f"S2 < T++ on {wins} of {len(results['ood_ppl'])}", "threshold": "≥ 3 of 4",
                 "verdict": yes(wins >= 3)})
    return rows


def q2_dir(step: int, data: str) -> Path:
    return RES / "q2" / f"s2_paper_step{step}_{data}_noise3"


def q3_dir(step: int, data: str) -> Path:
    return RES / "q3" / f"s2_paper_step{step}_{data}"


def load_json(d: Path) -> dict:
    return json.loads((d / "metrics.json").read_text())


def q2_arrays(step: int, data: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.load(q2_dir(step, data) / "arrays.npz")
    return a["e_final"], -a["logp"]


def collect() -> dict:
    s2_curve = val_curve(CSV / "s2-paper", "valid_final_step_loss")
    tpp_curve = val_curve(CSV / "tpp-paper", "valid_loss")
    s2_step, tpp_step = best_step(s2_curve), best_step(tpp_curve)
    q2 = {s: load_json(q2_dir(s2_step, s)) for s in SETS}
    q3 = {s: load_json(q3_dir(s2_step, s)) for s in SETS}
    assert all(q2[s]["config"]["tpp_step"] == tpp_step for s in SETS), "Q2 T++ reference is not T++'s best checkpoint"
    ce = {}  # (set, key) -> mean token CE
    for s in SETS:
        for z in (0, 3):
            for k in (2, 3):
                ce[s, f"k{k}_noise{z}"] = q3[s][f"part_a_noise{z}"]["ce_after_k_steps"][str(k)]
        e, c = q2_arrays(s2_step, s)
        for n in BON_N:
            ce[s, f"bon{n}"] = bon_ce(e, c, n)
        ce[s, "single"] = q2[s]["single_sample"]["ce"]
        ce[s, "tpp"] = q2[s]["single_sample"]["tpp_ce"]
    curve_steps = sorted(s2_curve)
    dyck, val = {}, {}
    for step in curve_steps:
        e, c = q2_arrays(step, "bigbench_dyck_languages")
        dyck[step * TOKENS_PER_STEP] = pct_improvement(float(c[0].mean()), bon_ce(e, c, 5))
        e, c = q2_arrays(step, "rp_val")
        val[step * TOKENS_PER_STEP] = pct_improvement(bon_ce(e, c, 2), bon_ce(e, c, 10))
    shift = {s: math.exp(ce[s, "single"] - ce["rp_val", "single"]) for s in SETS}
    keys = ["k2_noise3", "k3_noise3", "k2_noise0", "k3_noise0", *(f"bon{n}" for n in BON_N), "single", "tpp"]
    return {
        "s2_step": s2_step, "tpp_step": tpp_step, "s2_curve": s2_curve, "tpp_curve": tpp_curve,
        "gap": s2_curve[s2_step] - tpp_curve[tpp_step], "ce": ce, "q2": q2,
        "mean_ppl_all": {k: mean_ppl([ce[s, k] for s in OOD_SETS]) for k in keys},
        "mean_ppl": {"k2": mean_ppl([ce[s, "k2_noise3"] for s in OOD_SETS]),
                     "k3": mean_ppl([ce[s, "k3_noise3"] for s in OOD_SETS]),
                     "bon10": mean_ppl([ce[s, "bon10"] for s in OOD_SETS])},
        "dyck_curve": dyck, "val_curve": val, "shift": shift,
        "shift_points": [(shift[s], pct_improvement(ce[s, "k2_noise3"], ce[s, "bon10"])) for s in SETS],
        "ood_ppl": {s: (math.exp(ce[s, "single"]), math.exp(ce[s, "tpp"])) for s in OOD_SETS},
    }


def write_result(r: dict, rows: list[dict]) -> None:
    ce, L = r["ce"], []
    L += ["# Paper-scale results (generated by collect_paper.py)", "",
          f"Table 3 checkpoints (lowest validation loss): S2 step {r['s2_step']:,}, T++ step {r['tpp_step']:,}.", "",
          "## Pre-registered claims", "", "| # | Value | Supported if | Paper | Verdict |", "|---|---|---|---|---|"]
    L += [f"| {w['claim']} | {w['value']} | {w['threshold']} | {PAPER[w['claim']]} | **{w['verdict']}** |" for w in rows]
    L += ["", "## Perplexity per set (Table 3 analogue; S2 single sample, k = 3, σ = 3)", "",
          "| Set | Tokens | T++ ppl | S2 ppl | S2 BoN-10 ppl | OOD shift |", "|---|---|---|---|---|---|"]
    for s in SETS:
        L.append(f"| {s} | {r['q2'][s]['config']['n_tokens']:,} | {math.exp(ce[s, 'tpp']):.2f} | {math.exp(ce[s, 'single']):.2f} | "
                 f"{math.exp(ce[s, 'bon10']):.2f} | {r['shift'][s]:.2f} |")
    L += ["", "## Fig. 6a analogue (mean ppl over the 4 OOD sets)", "", "| Variant | Forward passes | Mean ppl |", "|---|---|---|"]
    names = [("tpp", "Transformer++", 1), ("k2_noise3", "No Thinking (k = 2, σ = 3)", 2),
             ("k3_noise3", "Thinking Longer (k = 3, σ = 3)", 3), *((f"bon{n}", f"Self-Verification BoN-{n}", 3 * n) for n in BON_N),
             ("k2_noise0", "k = 2, σ = 0 (secondary)", 2), ("k3_noise0", "k = 3, σ = 0 (secondary)", 3)]
    L += [f"| {label} | {fp} | {r['mean_ppl_all'][k]:.2f} |" for k, label, fp in names]
    L += ["", "## Validation curves and BoN gains over training", "",
          "| Step | Tokens (B) | T++ val CE | S2 val CE | Gap | Dyck BoN-5 vs 1 (%) | Val BoN-10 vs BoN-2 (%) |", "|---|---|---|---|---|---|---|"]
    for step in sorted(r["s2_curve"]):
        t = step * TOKENS_PER_STEP
        L.append(f"| {step:,} | {t / 1e9:.1f} | {r['tpp_curve'][step]:.4f} | {r['s2_curve'][step]:.4f} | "
                 f"{r['s2_curve'][step] - r['tpp_curve'][step]:+.4f} | {r['dyck_curve'][t]:+.2f} | {r['val_curve'][t]:+.2f} |")
    REPORT.mkdir(exist_ok=True)
    (REPORT / "result.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e1", lw=0.6)


def save(fig, name: str) -> None:
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(REPORT / "figures" / f"{name}.png", dpi=160)
    plt.close(fig)


def plot(r: dict) -> None:
    ppl = r["mean_ppl_all"]
    x_bon = [STEPS_PER_CANDIDATE * n for n in BON_N]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot([2, x_bon[-1]], [ppl["tpp"]] * 2, color=C_TPP, marker="o", ms=4, label="Transformer++ (1 pass)")
    ax.plot([2, 3, *x_bon], [ppl["k2_noise3"], ppl["k3_noise3"], *(ppl[f"bon{n}"] for n in BON_N)], color=C_EBT,
            marker="o", label="EBT: k = 2, k = 3, then BoN-2/5/10")
    ax.plot([2, 3], [ppl["k2_noise0"], ppl["k3_noise0"]], color=MUTED, ls=":", marker=".", label="EBT k = 2, 3 at σ = 0")
    for x, k in [(2, "k2_noise3"), (3, "k3_noise3"), *zip(x_bon, [f"bon{n}" for n in BON_N]), (x_bon[-1], "tpp")]:
        ax.annotate(f"{ppl[k]:.1f}", (x, ppl[k]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xticks([2, 3, *x_bon])
    style(ax, "Forward passes per token", "Mean ppl over 4 OOD sets", "Fig. 6a analogue: S2 vs T++ at 128M sequences (σ = 3)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, "fig6a")

    for name, curve, ylabel, title in [
            ("fig6b", r["dyck_curve"], "% ppl improvement, BoN-5 vs 1 sample", "Fig. 6b analogue: Dyck self-verification gain"),
            ("figB1a", r["val_curve"], "% ppl improvement, BoN-10 vs BoN-2", "Fig. B.1a analogue: validation BoN-10 vs BoN-2")]:
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot([t / 1e9 for t in curve], list(curve.values()), color=C_EBT, marker="o", ms=3)
        ax.axhline(0, color="k", lw=0.8)
        style(ax, "Training tokens (B)", ylabel, title)
        save(fig, name)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for s, (x, y) in zip(SETS, r["shift_points"]):
        ax.plot(x, y, "o", color=C_EBT)
        ax.annotate(s.replace("bigbench_", "BB "), (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    style(ax, "OOD shift = ppl(set) / ppl(validation)", "% ppl improvement, BoN-10 vs k = 2", "Fig. 7 analogue: thinking gain vs shift")
    save(fig, "fig7")

    steps = sorted(r["s2_curve"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    a1.plot([s / 1e3 for s in steps], [r["tpp_curve"][s] for s in steps], color=C_TPP, marker="o", ms=3, label="T++")
    a1.plot([s / 1e3 for s in steps], [r["s2_curve"][s] for s in steps], color=C_EBT, marker="o", ms=3, label="S2")
    a1.legend(fontsize=7, frameon=False)
    style(a1, "Step (k)", "Validation CE", "Validation CE")
    a2.plot([s / 1e3 for s in steps], [r["s2_curve"][s] - r["tpp_curve"][s] for s in steps], color="k", marker="o", ms=3)
    a2.axhline(0.149, color=MUTED, ls="--", lw=0.8)
    a2.axhline(0.114, color=MUTED, ls=":", lw=0.8)
    style(a2, "Step (k)", "S2 − T++ CE", "Gap (dashed: C1 threshold 0.149; dotted: 0.114)")
    save(fig, "gap_curve")
    plot_vs_paper(r)


def table3_ours(r: dict) -> dict:
    """Our Table 3 ppl in SETS order: pretrain = full validation at the Table 3 step; OOD = eval subset, S2 one sample."""
    ce = r["ce"]
    return {"tpp": [math.exp(r["tpp_curve"][r["tpp_step"]]), *(math.exp(ce[s, "tpp"]) for s in OOD_SETS)],
            "ebt": [math.exp(r["s2_curve"][r["s2_step"]]), *(math.exp(ce[s, "single"]) for s in OOD_SETS)]}


def plot_vs_paper(r: dict) -> None:
    """Slope chart per model: paper's Table 3 (hollow) -> ours (filled), one color per set, log ppl axis."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    ours = table3_ours(r)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8), sharey=True)
    for ax, model, title in zip(axes, ("tpp", "ebt"), ("Transformer++", "EBT (System 2)")):
        for s, c, p, o in zip(SETS, SET_COLORS, PAPER_TABLE3[model], ours[model]):
            ax.plot([0, 1], [p, o], color=c, lw=2, zorder=2)
            ax.plot(0, p, "o", ms=8, mfc="white", mec=c, mew=2, zorder=3)
            ax.plot(1, o, "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
            ax.annotate(f"{SET_NAMES[s]}  {100 * (o / p - 1):+.0f}%", (1, o), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=8, color="#0b0b0b")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([30, 40, 50, 70, 100, 140]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xticks([0, 1], ["Paper (Table 3)", "Ours"])
        ax.set_xlim(-0.25, 1.85)
        style(ax, "", "Perplexity (log scale)" if model == "tpp" else "", title)
        ax.grid(axis="x", visible=False)
    handles = [Line2D([], [], color=c, lw=2, marker="o", ms=6, label=SET_NAMES[s]) for s, c in zip(SETS, SET_COLORS)]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Table 3 perplexity: paper vs reproduction (xxs, 128M sequences; label = ours vs paper)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT / "figures" / "table3_vs_paper.png", dpi=160)
    plt.close(fig)


def main() -> None:
    r = collect()
    rows = verdicts(r)
    write_result(r, rows)
    plot(r)


if __name__ == "__main__":
    main()
