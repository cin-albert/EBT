"""Best-of-N self-verification curves from analyze_q2.py metrics.

Usage:
  python q2_verifier_quality/plot_q2.py   -> q2_verifier_quality/figures/bon_{curves,gain_over_random}.png  (xxs runs)
  python q2_verifier_quality/plot_q2.py --results DIR --out FILE.png \
      --panels RUN RUN --runs RUN RUN ... [--sequential]
    --panels: two runs drawn with all four pick rules; --runs: runs in the "gain over random" panel
    --sequential: color --runs as an ordered series (training steps) instead of categorically
    --no-sharey: independent y-axes for the two curve panels (when their CE ranges differ a lot)
    default: two standalone figures, OUT_STEM_curves.png (first --panels run only) and
             OUT_STEM_gain_over_random.png (--runs)
    --combined: one 3-panel figure at --out instead (two curve panels + gain panel)
  python q2_verifier_quality/plot_q2.py --results DIR --out FILE.png --grid RUN ... --titles TITLE ...
    --grid: one curve panel per run (e.g. one per OOD set), 2 columns, independent y-axes
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES, OUT = HERE / "results", HERE / "figures"

# categorical slots 1-3 of the dataviz reference palette + neutral; aqua (< 3:1 on white) is always dashed + labeled
C_LOW, C_HIGH, C_ORACLE, C_RAND = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"
RUNS = {  # dir -> label (order = legend order in the delta panel)
    "s2_step105000_noise3": "S2 105k, σ=3",
    "s2_step90000_noise3": "S2 90k, σ=3",
    "s2_step105000_noise0": "S2 105k, σ=0",
    "s1_step105000_noise0": "S1 105k, σ=0",
    "s1_step90000_noise0": "S1 90k, σ=0",
}
# panel 3 uses hues not used in panels 1-2 (violet, magenta) + grays for the flat sigma=0 runs
RUN_COLORS = ["#6250d6", "#e87ba4", "#52514e", "#8a8984", "#b5b4ae"]
RUN_STYLES = ["-", "--", "-", ":", "--"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
})


def label_of(run):
    """'s2_step105000_noise3' -> 'S2 105k, σ=3'; 's2_b64_16m_step250000_noise0' -> 'S2 250k, σ=0'."""
    m = re.match(r"(s\d).*step(\d+)_noise(\d+)", run)
    return f"{m[1].upper()} {int(m[2]) // 1000}k, σ={m[3]}" if m else run


def load(run):
    m = json.loads((RES / run / "metrics.json").read_text())
    bon = m["e_final"]["best_of_n"]
    n = np.array(sorted(int(k) for k in bon))
    get = lambda key: np.array([bon[str(k)][key] for k in n])
    return n, get, m


def curves_panel(ax, run, label):
    n, get, m = load(run)
    lines = [("ce_energy_pick", C_LOW, "-", "o", "lowest energy (BoN)"),
             ("ce_random", C_RAND, "-", None, "random"),
             ("ce_max_energy_pick", C_HIGH, "-", "s", "highest energy"),
             ("ce_oracle", C_ORACLE, "--", None, "oracle (lowest CE)")]
    for key, c, ls, mk, lab in lines:
        y = get(key)
        ax.plot(n, y, color=c, ls=ls, marker=mk, ms=4, label=lab)
        ax.text(n[-1] * 1.12, y[-1], f"{y[-1]:.3f}", color=INK, fontsize=7, va="center")
    tpp = m["single_sample"]["tpp_ce"]
    ax.axhline(tpp, color=INK, lw=1, ls=":")
    ax.text(1, tpp, " Transformer++", color=INK, fontsize=7, va="bottom")
    eff = get("verifier_efficiency")[-1]
    ax.set_title(f"{label}: CE vs N  (efficiency@{n[-1]} = {eff:+.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [str(k) for k in n])
    ax.set_xlim(0.85, n[-1] * 1.6)
    ax.set_xlabel("N candidates per token")
    ax.set_ylabel("mean CE of the picked candidate (nats)")


def delta_panel(ax, runs, sequential, title="All runs: gain over random"):
    if sequential:  # ordered checkpoints: violet ramp light -> dark for sigma>0, gray for sigma=0
        ramp = iter(["#c5bff2", "#a397ea", "#8273e0", "#6250d6", "#3f31a0", "#2a2070"])
        styles = [("#8a8984", ":") if run.endswith("noise0") else (next(ramp), "-") for run in runs]
    else:
        styles = list(zip(RUN_COLORS, RUN_STYLES))
    for run, (c, ls) in zip(runs, styles):
        label = RUNS.get(run, label_of(run))
        n, get, _ = load(run)
        ax.plot(n, get("ce_energy_pick") - get("ce_random"), color=c, ls=ls, marker="o", ms=4, label=label)
    ax.axhline(0, color=INK, lw=1)
    ax.text(32, -0.004, "= random pick ", color=INK, fontsize=7, va="top", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(n, [str(k) for k in n])
    ax.set_xlabel("N candidates per token")
    ax.set_ylabel("CE(lowest-E pick) − CE(random) \n> 0: verifier hurts")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")


def split_figures(args):
    stem = args.out.with_suffix("")
    run = args.panels[0]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    curves_panel(ax, run, RUNS.get(run, label_of(run)))
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.08))
    fig.tight_layout()
    path = Path(f"{stem}_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    delta_panel(ax, args.runs, args.sequential, title=args.gain_title)
    fig.tight_layout()
    path = Path(f"{stem}_gain_over_random.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def grid_figure(args):
    rows = (len(args.grid) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), squeeze=False)
    for ax, run, title in zip(axes.flat, args.grid, args.titles or args.grid):
        curves_panel(ax, run, title)
    for ax in axes.flat[len(args.grid):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out)
    print(f"wrote {args.out}")


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RES)
    ap.add_argument("--out", type=Path, default=OUT / "bon.png")
    ap.add_argument("--panels", nargs="+", default=["s2_step105000_noise3", "s2_step90000_noise3"])
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--no-sharey", dest="sharey", action="store_false", help="independent y-axes for the two panels")
    ap.add_argument("--title", default="Q2  Best-of-N by lowest energy: helps only at N = 2, then hurts; "
                                       "without Langevin noise (σ = 0) there is nothing to choose")
    ap.add_argument("--combined", action="store_true")
    ap.add_argument("--grid", nargs="+", help="one curve panel per run")
    ap.add_argument("--titles", nargs="+", help="panel titles for --grid (default: run names)")
    ap.add_argument("--gain-title", default="Lowest-energy BoN vs a random pick, by run\n"
                    "(xxs, 105k / 90k; above 0 = picking by energy is worse than random)")
    args = ap.parse_args()
    RES = args.results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.grid:
        grid_figure(args)
        return
    if not args.combined:
        split_figures(args)
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, run in zip(axes[:2], args.panels):
        curves_panel(ax, run, RUNS.get(run, label_of(run)))
    if args.sharey:
        axes[1].sharey(axes[0])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.36, 0))
    delta_panel(axes[2], args.runs, args.sequential)
    fig.suptitle(args.title, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
