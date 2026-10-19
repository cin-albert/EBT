"""Tables for report/result.md: validation curves and S2-T++ gaps per data point, plus Q2/Q3 headline metrics."""
import glob
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent
REPRO = WS.parent / "reproduce_xxs_scaling"
TS = WS.parent / "thinking_study"
sys.path.insert(0, str(REPRO))
from collect_results import comparable_loss, read_valid_rows  # noqa: E402

POINTS = {  # label -> (T++ csv dir, S2 csv dir, Q2 dir σ=3, Q3 dir)
    "3.36M seqs, b32 (105k)": (REPRO / "runs/logs/csv/baseline-xxs-scaling-105k", REPRO / "runs/logs/csv/ebt-s2-xxs-scaling-105k",
                              TS / "q2_verifier_quality/results/s2_step105000_noise3", TS / "q3_energy_landscape/results/s2_step105000"),
    "16M seqs, b64 (250k)": (WS / "runs/logs/csv/tpp-b64-16m", WS / "runs/logs/csv/s2-b64-16m",
                            WS / "results/q2_verifier_quality/s2_b64_16m_step250000_noise3", WS / "results/q3_energy_landscape/s2_b64_16m_step250000"),
}
CURVE = WS / "results/q2_verifier_quality"  # s2_b64_16m_step<N>_noise3 along the 16M run


def curve(run_dir: Path) -> dict[int, float]:
    """Validation loss per logged step; on duplicates keep the first, since a resumed run re-validates its restart step."""
    out: dict[int, float] = {}
    for csv_path in sorted(glob.glob(str(run_dir / "version_*/metrics.csv"))):
        for r in read_valid_rows([csv_path]):
            out.setdefault(r["step"], comparable_loss(r))
    return out


def point_row(label: str, tpp_dir: Path, s2_dir: Path, q2_dir: Path, q3_dir: Path) -> str:
    tpp, s2 = curve(tpp_dir), curve(s2_dir)
    last = max(s2)
    q2 = json.loads((q2_dir / "metrics.json").read_text())["e_final"]
    q3 = json.loads((q3_dir / "metrics.json").read_text())["part_a_noise0"]
    ce_k = q3["ce_after_k_steps"]
    return (f"| {label} | {tpp[last]:.4f} | {s2[last]:.4f} | {s2[last] - tpp[last]:+.4f} | "
            f"{q2['best_of_n']['32']['verifier_efficiency']:+.2f} | {q2['pairwise_consistency_gap_gt_0.1'][0]:.3f} | "
            f"{q3['best_k']['k']} | {q3['best_k']['ce_at_train_steps']:.3f} → {ce_k[str(max(map(int, ce_k)))]:.3f} |")


def main() -> None:
    lines = ["## Final validation loss and gap", "",
             "| point | T++ | S2 | gap (S2 − T++) | Q2 BoN-32 efficiency (σ=3) | Q2 consistency (\\|ΔCE\\|>0.1) | Q3 best k | Q3 CE trained k → k=32 |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [point_row(label, *dirs) for label, dirs in POINTS.items()]
    tpp, s2 = (curve(d) for d in POINTS["16M seqs, b64 (250k)"][:2])
    lines += ["", "## Gap along the 16M run", "", "| step | T++ | S2 | gap | Q2 BoN-32 efficiency (σ=3) | Q2 consistency |",
              "|---|---|---|---|---|---|"]
    for k in sorted(set(tpp) & set(s2)):  # CSV logs validation at global step - 1
        m = CURVE / f"s2_b64_16m_step{k + 1}_noise3/metrics.json"
        q2 = json.loads(m.read_text())["e_final"] if m.exists() else None
        q2_cells = f"{q2['best_of_n']['32']['verifier_efficiency']:+.2f} | {q2['pairwise_consistency_gap_gt_0.1'][0]:.3f}" if q2 else "– | –"
        lines.append(f"| {k + 1} | {tpp[k]:.4f} | {s2[k]:.4f} | {s2[k] - tpp[k]:+.4f} | {q2_cells} |")
    out = WS / "report/result.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


if __name__ == "__main__":
    main()
