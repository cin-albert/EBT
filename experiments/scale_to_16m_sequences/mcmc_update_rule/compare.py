"""H1-H4 table and CE-vs-k figure: probs baseline (S2 105k/b32, noise 3) vs the update-rule runs (noise 0).

H3 is read at each run's training noise (baseline 3, rule runs 0): at noise 0 the probs baseline's candidates are
near-identical (Q2 diversity: mean JS 8.5e-7), so its noise-0 best-of-N says nothing. A noise level a run's Q3 did not
cover shows as "—".

Usage: python compare.py   -> report/compare.md, report/figures/ce_vs_k.png
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WS = Path(__file__).resolve().parent
EXP = WS.parents[1]
TS = EXP / "thinking_study"
LP_CSV = WS / "runs/logs/csv/s2-rule-log-probs-105k/version_0/metrics.csv"
RUNS = {  # label -> (metrics.csv, checkpoint step, Q2 dir prefix, Q3 dir, training noise)
    "probs (baseline, noise 3) @105k": (EXP / "reproduce_xxs_scaling/runs/logs/csv/ebt-s2-xxs-scaling-105k/version_0/metrics.csv", 105000,
                                        TS / "q2_verifier_quality/results/s2_step105000", TS / "q3_energy_landscape/results/s2_step105000", 3),
    "logits (noise 0) @105k": (WS / "runs/logs/csv/s2-rule-logits-105k/version_0/metrics.csv", 105000,
                               WS / "results/q2/s2_rule_logits_step105000", WS / "results/q3/s2_rule_logits_step105000", 0),
    "log_probs (noise 0) @105k": (LP_CSV, 105000,
                                  WS / "results/q2/s2_rule_log_probs_step105000", WS / "results/q3/s2_rule_log_probs_step105000", 0),
    # supplementary: best validation checkpoint, before the run became unstable (report §3)
    "log_probs (noise 0) @30k": (LP_CSV, 30000,
                                 WS / "results/q2/s2_rule_log_probs_step30000", WS / "results/q3/s2_rule_log_probs_step30000", 0),
}


def valid_loss_at(csv, step: int) -> float:
    """Validation loss of the checkpoint saved at `step` (Lightning logs it at step - 1)."""
    df = pd.read_csv(csv)
    return float(df[df["step"] < step]["valid_loss"].dropna().iloc[-1])


def best_k(q3: dict, noise: int) -> tuple[int, float] | None:
    part = q3.get(f"part_a_noise{noise}")
    if not part:
        return None
    return int(part["best_k"]["k"]), float(part["best_k"]["ce"])


def bon_deltas(q2: dict) -> dict[int, float]:
    return {int(n): round(v["ce_energy_pick"] - v["ce_random"], 6) for n, v in q2["e_final"]["best_of_n"].items() if n != "1"}


def gnorm_ratio(q3_extra: dict, noise: int) -> float | None:
    part = q3_extra.get(f"part_a_noise{noise}")
    if not part:
        return None
    return float(part["gradient"]["ratio_last_over_trained"]["p50"])


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def _k(b) -> str:
    return "—" if b is None else f"{b[0]} ({b[1]:.4f})"


def _r(x) -> str:
    return "—" if x is None else f"{x:.2f}"


def main():
    present = {k: v for k, v in RUNS.items() if v[0].exists() and (v[3] / "metrics.json").exists()}
    lines = ["| Rule | H1 valid loss | H2 best k (CE), noise 0 | H2 best k (CE), noise 3 "
             "| H3 energy − random, N = 2 / 4 / 8 / 16 / 32 (training noise) | H4 median ‖d₃₁‖/‖d₂‖, noise 0 / 3 |",
             "|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for label, (csv, step, q2_prefix, q3_dir, train_noise) in present.items():
        q3, extra = load(q3_dir / "metrics.json"), load(q3_dir / "metrics_extra.json")
        d = bon_deltas(load(Path(f"{q2_prefix}_noise{train_noise}") / "metrics.json"))
        lines.append(f"| {label} | {valid_loss_at(csv, step):.4f} | {_k(best_k(q3, 0))} | {_k(best_k(q3, 3))} | "
                     + " / ".join(f"{d[n]:+.3f}" for n in sorted(d))
                     + f" | {_r(gnorm_ratio(extra, 0))} / {_r(gnorm_ratio(extra, 3))} |")
        for ax, noise in zip(axes, (0, 3)):
            part = q3.get(f"part_a_noise{noise}")
            if part:
                ce = part["ce_after_k_steps"]
                ks = sorted(map(int, ce))
                ax.plot(ks, [ce[str(k)] for k in ks], marker=".", label=label)
    for ax, noise in zip(axes, (0, 3)):
        ax.axvline(3, color="#8a8984", ls=":", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("MCMC steps k")
        ax.set_title(f"rp_val CE after k steps (inference noise {noise})", fontsize=10)
        ax.grid(color="#e6e5e1", lw=0.6)
    axes[0].set_ylabel("CE (nats, log scale)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    (WS / "report/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(WS / "report/figures/ce_vs_k.png", dpi=160)
    (WS / "report/compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
