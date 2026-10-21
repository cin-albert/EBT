"""Paper Fig. 6a analogue: mean perplexity over the four OOD sets vs number of forward passes.

EBT No Thinking = k = 2 steps, Thinking Longer = k = 3 (S2 trains with 2-3 steps), both from Q3 Part A.
Self-Verification = lowest-energy pick among N candidates of 3 steps each (forward passes 3N, N = 2, 5, 10),
recomputed from the Q2 sigma = 3 arrays. Transformer++ does one pass per token, drawn flat for reference.
y = mean over sets of per-set ppl = exp(mean token CE); noise 3 throughout, noise 0 as a dotted reference.

Usage: python ood/plot_fig6a.py   -> report/figures/fig6a_ood_thinking.png, table on stdout
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OOD = Path(__file__).resolve().parent
WS = OOD.parent
SETS = ["gsm8k", "squad", "bigbench_elementary_math_qa", "bigbench_dyck_languages"]
RUN = "s2_b64_16m_step250000"
BON_N = (2, 5, 10)
STEPS = 3  # per candidate in Q2
C_TPP, C_EBT, MUTED = "#eb6834", "#2a78d6", "#8a8984"


def bon_ce(e: np.ndarray, ce: np.ndarray, n: int) -> float:
    """Mean CE of the lowest-energy candidate among the first n; e, ce are (candidates, tokens)."""
    pick = e[:n].argmin(0)
    return float(np.take_along_axis(ce[:n], pick[None], 0).mean())


def mean_ppl(ces) -> float:
    return float(np.mean(np.exp(ces)))


def per_set(s: str) -> dict:
    q3 = json.loads((WS / f"results/ood/q3_energy_landscape/{RUN}_{s}/metrics.json").read_text())
    a = np.load(WS / f"results/ood/q2_verifier_quality/{RUN}_{s}_noise3/arrays.npz")
    ce = -a["logp"]
    row = {f"k{k}_noise{z}": q3[f"part_a_noise{z}"]["ce_after_k_steps"][str(k)] for k in (2, 3) for z in (0, 3)}
    row |= {f"bon{n}": bon_ce(a["e_final"], ce, n) for n in BON_N}
    row["tpp"] = float(-a["tpp_logp"].mean())
    return row


def main():
    rows = {s: per_set(s) for s in SETS}
    ppl = {key: mean_ppl([rows[s][key] for s in SETS]) for key in rows[SETS[0]]}
    print("| key | " + " | ".join(SETS) + " | mean ppl |\n|---|" + "---|" * (len(SETS) + 1))
    for key in ppl:
        print(f"| {key} | " + " | ".join(f"{np.exp(rows[s][key]):.2f}" for s in SETS) + f" | {ppl[key]:.2f} |")

    x_bon = [STEPS * n for n in BON_N]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot([2, x_bon[-1]], [ppl["tpp"]] * 2, color=C_TPP, marker="o", ms=4, label="Transformer++ (1 pass)")
    ax.plot([2, 3], [ppl["k2_noise3"], ppl["k3_noise3"]], color=C_EBT)
    ax.plot([3, x_bon[0]], [ppl["k3_noise3"], ppl["bon2"]], color=C_EBT)
    ax.plot([2], [ppl["k2_noise3"]], color=C_EBT, marker="o", ls="", label="EBT No Thinking (k = 2)")
    ax.plot([3], [ppl["k3_noise3"]], color=C_EBT, marker="*", ms=9, ls="", label="EBT Thinking Longer (k = 3)")
    ax.plot(x_bon, [ppl[f"bon{n}"] for n in BON_N], color=C_EBT, marker="^", label="EBT Self-Verification (BoN-N, k = 3)")
    ax.plot([2, 3], [ppl["k2_noise0"], ppl["k3_noise0"]], color=MUTED, ls=":", marker=".", label="EBT k = 2, 3 at noise 0")
    for x, key in [(2, "k2_noise3"), (3, "k3_noise3"), *zip(x_bon, [f"bon{n}" for n in BON_N])]:
        ax.annotate(f"{ppl[key]:.1f}", (x, ppl[key]), textcoords="offset points", xytext=(6, 4), fontsize=7)
    for x, key in [(3, "k3_noise0"), (x_bon[-1], "tpp")]:
        ax.annotate(f"{ppl[key]:.1f}", (x, ppl[key]), textcoords="offset points", xytext=(6, 4), fontsize=7, color=MUTED)
    ax.set_xticks([2, 3, *x_bon])
    ax.set_xlabel("Number of forward passes per token")
    ax.set_ylabel("Mean perplexity over 4 OOD sets (lower is better)")
    ax.set_title("S2 vs Transformer++ 250k/b64 on OOD data (σ = 3)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e1", lw=0.6)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = OOD / "report/figures/fig6a_ood_thinking.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
