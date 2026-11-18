"""Energy of the S2 step-1M prediction over inference GD steps t = 0..31 (Q3 Part A), per set and noise.

Each token's energy trajectory is min-max scaled to [0, 1] over its 32 steps, so tokens with different energy
magnitudes count equally; tokens whose energy never changes are dropped. Line = mean over tokens, band = IQR.
Usage: python plot_energy_steps.py   (writes figures/energy_steps.png; not part of the report)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collect_paper import SET_COLORS, SET_NAMES, SETS, style

WS = Path(__file__).resolve().parent
Q3 = WS / "results/q3"
TRAIN_STEPS = 3  # the step count S2 is trained and evaluated with


def minmax_per_token(energy: np.ndarray) -> np.ndarray:
    """(steps, tokens) -> each column scaled to [0, 1]; columns with zero range are dropped."""
    lo, hi = energy.min(0), energy.max(0)
    keep = hi > lo
    return (energy[:, keep] - lo[keep]) / (hi[keep] - lo[keep])


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, noise, title in zip(axes, (0, 3), ("Noise σ = 0", "Noise σ = 3 (training / test setting)")):
        for s, c in zip(SETS, SET_COLORS):
            e = minmax_per_token(np.load(Q3 / f"s2_paper_step1000000_{s}/part_a_noise{noise}.npz")["energy"])
            t = np.arange(e.shape[0])
            ax.plot(t, e.mean(1), color=c, lw=2, label=SET_NAMES[s])
            if s == "rp_val":
                ax.fill_between(t, *np.percentile(e, [25, 75], axis=1), color=c, alpha=0.15, lw=0)
        ax.axvline(TRAIN_STEPS - 1, color="#898781", ls="--", lw=0.8)
        ax.annotate("last step at k = 3 (training / test)", (TRAIN_STEPS - 1, 1.0), xytext=(4, -2), textcoords="offset points",
                    fontsize=7, color="#52514e", va="top")
        ax.set_ylim(-0.02, 1.02)
        style(ax, "GD step t", "Energy, min-max scaled per token" if noise == 0 else "", title)
    axes[0].legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle("S2 (step 1M): energy over inference GD steps, mean over tokens (band: IQR on validation)",
                 fontsize=10)
    fig.tight_layout()
    (WS / "figures").mkdir(exist_ok=True)
    fig.savefig(WS / "figures/energy_steps.png", dpi=160)


if __name__ == "__main__":
    main()
