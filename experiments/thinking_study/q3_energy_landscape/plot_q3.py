"""Energy-landscape figures from saved Q2/Q3 arrays (charts 2.1-2.6 of ENERGY_LANDSCAPE_VISUALIZATIONS.md).

Offline only. Charts 2.2B and 2.3C need a Part B re-run and are not drawn here.

Usage: python q3_energy_landscape/plot_q3.py [--set thinking|scale16m]
  thinking (default): S1 vs S2 at 105k             -> q3_energy_landscape/figures/*.png
  scale16m:           S2 105k/b32 vs S2 250k/b64   -> scale_to_16m_sequences/report/figures/*.png
"""
import argparse
import json
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
Q3 = HERE / "results"
Q2 = HERE.parent / "q2_verifier_quality" / "results"
SCALE = HERE.parents[1] / "scale_to_16m_sequences"

# categorical slots 1-3 of the dataviz reference palette; slot 3 (aqua) is < 3:1 on white -> always dashed + labeled
C_S1, C_S2, C_S2N = "#2a78d6", "#eb6834", "#1baf7a"
STATE_RAMP = ["#86b6ef", "#2a78d6", "#104281"]  # sequential blue: t=0 -> trained -> t=31
INK, MUTED = "#0b0b0b", "#52514e"


class Model(NamedTuple):
    q3: Path  # run_q3.py output dir
    q2: Path  # run_q2.py output dir prefix; "_noise0" / "_noise3" appended
    alpha: float
    tr: int  # last trained state index
    color: str
    ce_ylim: tuple | None  # chart 2.6 CE axis; S2's k=1 is clipped to show the odd/even alternation
    h_ylim: tuple  # chart 2.6 entropy axis


S2_105K = Model(Q3 / "s2_step105000", Q2 / "s2_step105000", 5.0, 2, C_S2, (3.9, 4.35), (0, 6.5))
SETS = {  # name -> (models, output dir, add the thinking-study findings to the titles)
    "thinking": ({"S1": Model(Q3 / "s1_step105000", Q2 / "s1_step105000", 528.926, 1, C_S1, None, (0, 5)),
                  "S2": S2_105K}, HERE / "figures", True),
    "scale16m": ({"S2 105k/b32": S2_105K._replace(color=C_S1, ce_ylim=(3.6, 4.35)),
                  "S2 250k/b64": Model(SCALE / "results/q3_energy_landscape/s2_b64_16m_step250000",
                                       SCALE / "results/q2_verifier_quality/s2_b64_16m_step250000",
                                       5.0, 2, C_S2, (3.6, 4.35), (0, 6.5))},
                 SCALE / "report" / "figures", False),
}
MODELS, OUT, FINDINGS = SETS["thinking"]

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
})


def title(what: str, with_finding: str) -> str:
    """Chart title; findings were read off the thinking-study S1/S2 figures, so other sets get `what` only."""
    return with_finding if FINDINGS else what


def band(ax, x, y, color, label, ls="-"):
    """Median line + p10-p90 band over tokens; y is (steps, tokens)."""
    lo, med, hi = np.nanpercentile(y, [10, 50, 90], axis=1)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    ax.plot(x, med, color=color, ls=ls, label=label)


def mark_trained(ax, k, label=True):
    ax.axvline(k, color=MUTED, lw=1, ls=":")
    if label:
        ax.text(k, ax.get_ylim()[1], " trained", color=MUTED, fontsize=7, va="top")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def load_a(mod, noise=0):
    return np.load(mod.q3 / f"part_a_noise{noise}.npz")


def chart_thinking_longer():
    """2.6: CE after k steps and per-state E, Phi, entropy (means), one column per model (S1/S2 scales differ ~100x)."""
    rows = [("CE after k steps (nats)", "ce"), ("Energy E", "energy"), ("Φ = αE − H − ½‖p‖²", "phi"),
            ("Entropy H(p) (nats)", "entropy")]
    fig, axes = plt.subplots(4, len(MODELS), figsize=(4 * len(MODELS), 8.5), sharex=True)
    for j, (name, mod) in enumerate(MODELS.items()):
        metrics = json.loads((mod.q3 / "metrics.json").read_text())
        has_n3 = "part_a_noise3" in metrics
        series = [(0, mod.color, "-", "noise 0")] + ([(3, C_S2N, "--", "noise 3")] if has_n3 else [])
        for noise, c, ls, lab in series:
            m = metrics[f"part_a_noise{noise}"]
            for i, (_, key) in enumerate(rows):
                ax = axes[i, j]
                if key == "ce":  # x = k steps taken
                    d = m["ce_after_k_steps"]
                    x, y = np.array([int(k) for k in d]), np.array(list(d.values()))
                else:  # x = state index t (state after t steps)
                    d = m["mean_per_step"][key]
                    x, y = np.array([int(k) for k in d]), np.array(list(d.values()))
                ax.plot(x, y, color=c, ls=ls, lw=1.6, label=lab)
                if key == "ce":
                    kb = x[np.argmin(y)]
                    ax.plot(kb, y.min(), "o", ms=6, color=c, mec="white", mew=1.5)
                    ax.annotate(f"best k={kb}: {y.min():.3f} ({lab})" if has_n3 else f"best k={kb}: {y.min():.3f}",
                                (kb, y.min()), xytext=(14, -10 if noise == 0 else 18),
                                textcoords="offset points", fontsize=7, color=INK)
        for i, (lab, key) in enumerate(rows):
            ax = axes[i, j]
            if key == "ce" and mod.ce_ylim:
                ax.set_ylim(mod.ce_ylim)
            if key == "entropy":
                ax.set_ylim(mod.h_ylim)
            mark_trained(ax, mod.tr + 1, label=(i == 0))  # k = tr+1 steps; state index t = tr+1 is the returned prediction
            if j == 0:
                ax.set_ylabel(lab)
        axes[0, j].set_title(f"{name} (α = {mod.alpha:g})")
        axes[-1, j].set_xlabel("k (steps) / t (state index)")
        if has_n3:
            axes[0, j].legend(loc="upper right", fontsize=7)
    fig.suptitle(title("2.6  Thinking longer", "2.6  Thinking longer: loss is best at the trained step count; S2 alternates odd/even"),
                 fontsize=10)
    save(fig, "2_6_thinking_longer.png")


def chart_slope():
    """2.1: alpha * ||g_p|| by state; same scale for both models after multiplying by alpha."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, mod in MODELS.items():
        g = load_a(mod)["gnorm"]
        band(ax, np.arange(g.shape[0]), mod.alpha * g, mod.color, name)
        ax.axvline(mod.tr, color=mod.color, lw=1, ls=":")
        ax.text(31.3, mod.alpha * np.median(g[-1]), name, color=INK, fontsize=8, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("state index t")
    ax.set_ylabel("α · ‖∇_p E‖ (median, p10–p90)")
    ax.set_title(title("2.1  Gradient size", "2.1  Gradient size: no vanishing / exploding, no decay at convergence") +
                 "\n(dotted = trained state; p ∈ [0, 1], so each step is a jump)", fontsize=9)
    ax.legend(loc="upper right")
    save(fig, "2_1_gradient_norm.png")


def chart_cos():
    """2.2A: cos(g_t, g_{t-1}) by state."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for name, mod in MODELS.items():
        c = load_a(mod)["cos_prev"][1:]
        band(ax, np.arange(1, c.shape[0] + 1), c, mod.color, name)
    ax.set_ylim(0.9, 1.005)
    ax.set_xlabel("state index t")
    ax.set_ylabel("cos(g_t, g_{t−1})")
    ax.set_title(title("2.2A  Gradient direction across steps", "2.2A  Gradient direction is almost constant after t = 1\n"
                       "(fixed-point signature, not a straight path)"), fontsize=9)
    ax.legend(loc="lower right")
    save(fig, "2_2a_gradient_cos.png")


def part_b(mod):
    b = np.load(mod.q3 / "part_b.npz")
    states = sorted({int(k[1:].split("_")[0]) for k in b.files if k.startswith("t")})
    return b, states


def chart_rho():
    """2.3A: distribution of the spectral radius of the update-map Jacobian at three states."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 3.2), sharey=True)
    bins = np.linspace(-3, 1.5, 60)
    for ax, (name, mod) in zip(axes, MODELS.items()):
        b, states = part_b(mod)
        for c, t in zip(STATE_RAMP, states):
            r = np.log10(np.clip(b[f"t{t}_rho"], 1e-3, None))
            lab = f"t={t}" + (" (trained)" if t == mod.tr else "") + f"  median {np.median(b[f't{t}_rho']):.2f}"
            ax.hist(r, bins=bins, histtype="step", lw=2, color=c, density=True, label=lab)
        ax.axvline(0, color=INK, lw=1)
        ax.text(0.03, ax.get_ylim()[1] * 0.95, "ρ = 1", fontsize=7, color=INK, va="top")
        ax.set_title(name)
        ax.set_xlabel("log10 ρ(J_T)   (< 0: contracting)")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("density over tokens")
    fig.suptitle(title("2.3A  Stability of the update map", "2.3A  Stability of the update map: S2 expands at t = 0, both contract later"), fontsize=10)
    save(fig, "2_3a_rho.png")


def chart_spectrum():
    """2.3B: pooled density of lambda / lambda_max of the per-token Hessian of E in e-space."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 3.2), sharey=True)
    bins = np.linspace(-3, 1.05, 82)
    for ax, (name, mod) in zip(axes, MODELS.items()):
        b, states = part_b(mod)
        for c, t in zip(STATE_RAMP, states):
            ev = b[f"t{t}_eig"]
            r = np.clip(ev / ev[:, -1:], -3, None).ravel()
            neg = (ev < 0).mean()
            ax.hist(r, bins=bins, histtype="step", lw=2, color=c, density=True,
                    label=f"t={t}" + (" (trained)" if t == mod.tr else "") + f"  neg {neg:.0%}")
        ax.axvline(0, color=INK, lw=1)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("λ / λ_max  (clipped at −3)")
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("density (log)")
    fig.suptitle(title("2.3B  Hessian of E", "2.3B  Hessian of E: no convex token; S2 moves from concave to a valley, "
                                             "S1 stays a balanced saddle"), fontsize=10)
    save(fig, "2_3b_hessian_spectrum.png")


def chart_line_slices():
    """2.4: E, Phi, CE along z = p - s*alpha*grad, min-max normalized per token."""
    names = [("E", "line_energy", C_S1, "-"), ("Φ", "line_phi", C_S2, "-"), ("CE", "line_logp", C_S2N, "--")]
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(9, 2.5 * len(MODELS)), sharex=True, sharey=True)
    for i, (name, mod) in enumerate(MODELS.items()):
        b, states = part_b(mod)
        s = b["line_s"]
        for j, t in enumerate(states):
            ax = axes[i, j]
            for lab, key, c, ls in names:
                y = b[f"t{t}_{key}"] * (-1 if key == "line_logp" else 1)  # logp -> CE
                lo, hi = y.min(0, keepdims=True), y.max(0, keepdims=True)
                band(ax, s, (y - lo) / (hi - lo + 1e-12), c, lab, ls=ls)
                k = np.argmin(np.median((y - lo) / (hi - lo + 1e-12), 1))
                ax.plot(s[k], 0, "v", color=c, ms=7, clip_on=False)
            ax.axvline(1, color=MUTED, lw=1, ls=":")
            ax.set_title(f"{name}, t={t}" + (" (trained)" if t == mod.tr else ""))
            if i == len(MODELS) - 1:
                ax.set_xlabel("step scale s  (s = 1: actual update)")
            if j == 0:
                ax.set_ylabel("per-token min–max\nnormalized value")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(title("2.4  Along the update (▼ = median argmin)", "2.4  Along the update: Φ is minimized at s = 1, "
                       "CE at s ≈ 1.25–1.75, E keeps falling (▼ = median argmin)"), fontsize=10)
    save(fig, "2_4_line_slices.png")


def chart_energy_entropy():
    """2.5: across tokens (Q2 sigma=0 candidate 0) per model, then within token (Q2 sigma=3 candidate pairs) per model with sigma=3 arrays."""
    within = [(name, mod) for name, mod in MODELS.items() if Path(f"{mod.q2}_noise3").is_dir()]
    fig, axes = plt.subplots(1, len(MODELS) + len(within), figsize=(11 / 3 * (len(MODELS) + len(within)), 3.6))
    for ax, (name, mod) in zip(axes, MODELS.items()):
        a = np.load(f"{mod.q2}_noise0/arrays.npz")
        e, h, ce = a["e_final"][0], a["entropy"][0], -a["logp"][0]
        hb = ax.hexbin(h, e, C=ce, reduce_C_function=np.mean, gridsize=45, cmap="Blues", mincnt=20,
                       vmin=0, vmax=10, linewidths=0)
        rank = lambda x: np.argsort(np.argsort(x))
        rho = np.corrcoef(rank(e), rank(h))[0, 1]
        ax.set_title(f"{name}: across tokens (Spearman E–H = {rho:.2f})", fontsize=9)
        ax.set_xlabel("entropy H of the prediction (nats)")
        ax.set_ylabel("energy E (final)")
        ax.set_ylim(np.percentile(e, [0.1, 99.9]))
        fig.colorbar(hb, ax=ax, label="mean CE in bin (nats)")
    for ax, (name, mod) in zip(axes[len(MODELS):], within):
        a = np.load(f"{mod.q2}_noise3/arrays.npz")
        rng = np.random.default_rng(0)
        tok = rng.choice(a["e_final"].shape[1], 20000, replace=False)
        dE = a["e_final"][0, tok] - a["e_final"][1, tok]
        dH = a["entropy"][0, tok] - a["entropy"][1, tok]
        dCE = -(a["logp"][0, tok] - a["logp"][1, tok])
        keep = (np.abs(dCE) > 0.1) & (dE != 0)
        agree = np.sign(dE) == np.sign(dCE)  # True: the lower-energy candidate also has the lower CE
        order = rng.permutation(np.flatnonzero(keep))  # random draw order so neither color hides the other
        ax.scatter(dE[order], dH[order], s=3, alpha=0.35, lw=0, rasterized=True,
                   c=np.where(agree[order], C_S1, C_S2))
        for c, lab in [(C_S1, f"lower-E candidate has lower CE ({agree[keep].mean():.0%})"),
                       (C_S2, f"lower-E candidate has higher CE ({1 - agree[keep].mean():.0%})")]:
            ax.scatter([], [], s=12, color=c, label=lab)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        same = np.mean(np.sign(dE) == np.sign(dH))
        ax.set_title(f"{name} σ=3: within token, candidate pairs\n(|ΔCE| > 0.1) sign(ΔE) = sign(ΔH) in {same:.0%} of pairs",
                     fontsize=9)
        ax.set_xlabel("ΔE (candidate 0 − candidate 1)")
        ax.set_ylabel("ΔH (nats)")
        ax.set_xlim(np.percentile(dE, [0.5, 99.5]))
        ax.set_ylim(np.percentile(dH, [0.5, 99.5]))
        ax.legend(fontsize=7, loc="upper left", markerscale=1)
    fig.suptitle(title("2.5  Energy vs entropy", "2.5  Energy vs entropy: weak across tokens, near-deterministic "
                       "within a token (sharpness bias)"), fontsize=10)
    save(fig, "2_5_energy_entropy.png")


def main():
    global MODELS, OUT, FINDINGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, default="thinking")
    MODELS, OUT, FINDINGS = SETS[ap.parse_args().set]
    OUT.mkdir(parents=True, exist_ok=True)
    chart_thinking_longer()
    chart_slope()
    chart_cos()
    chart_rho()
    chart_spectrum()
    chart_line_slices()
    chart_energy_entropy()


if __name__ == "__main__":
    main()
