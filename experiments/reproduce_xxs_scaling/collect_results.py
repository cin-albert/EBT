"""Build report/result.md (headings and tables only) from CSVLogger metrics and benchmark JSON.

Comparable loss: EBT -> valid_final_step_loss (prediction after the last MCMC step; valid_loss
averages all steps); Transformer++ -> valid_loss. Perplexity is exp(comparable loss), because the
logged valid_perplexity is a mean of per-batch exp(loss).
"""
import argparse
import csv
import glob
import json
import math

VALID_KEYS = ("valid_loss", "valid_final_step_loss", "valid_perplexity")
BENCH_KEYS = ("steps_per_s", "tokens_per_s", "peak_mem_mib", "eta_hours_105k")


def read_valid_rows(metrics_csvs):
    rows = {}
    for path in metrics_csvs:
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("valid_loss"):
                    step = int(row["step"])
                    rows[step] = {"step": step, **{k: float(row[k]) for k in VALID_KEYS if row.get(k)}}
    return [rows[step] for step in sorted(rows)]


def comparable_loss(row):
    return row.get("valid_final_step_loss", row["valid_loss"])


def perplexity(loss):
    return math.exp(loss) if loss < 700 else math.inf  # exp overflows a float above ~709


def summary_table(curves, benchmarks):
    lines = [
        "| model | step | valid loss | valid ppl | steps/s | tokens/s | peak mem (MiB) | ETA 105k (h) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, rows in curves.items():
        last = rows[-1]
        loss = comparable_loss(last)
        bench = benchmarks.get(label, {})
        cells = [str(bench[k]) if bench.get(k) is not None else "–" for k in BENCH_KEYS]
        lines.append(f"| {label} | {last['step']} | {loss:.4f} | {perplexity(loss):.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def curve_table(curves):
    labels = list(curves)
    by_step = {label: {r["step"]: r for r in rows} for label, rows in curves.items()}
    steps = sorted({step for rows in by_step.values() for step in rows})
    header = "| step | " + " | ".join(f"{l} loss | {l} ppl" for l in labels) + " |"
    lines = [header, "|---|" + "---|---|" * len(labels)]
    for step in steps:
        cells = []
        for label in labels:
            row = by_step[label].get(step)
            if row is None:
                cells += ["–", "–"]
            else:
                loss = comparable_loss(row)
                cells += [f"{loss:.4f}", f"{perplexity(loss):.2f}"]
        lines.append(f"| {step} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_result_md(curves, benchmarks):
    return "\n".join([
        "# EBT vs Transformer++ — xxs scaling run (numerical results)",
        "",
        "## Final validation and throughput",
        "",
        summary_table(curves, benchmarks),
        "",
        "## Validation curve",
        "",
        curve_table(curves),
        "",
    ])


def _pairs(values):
    return [value.split("=", 1) for value in values]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help='LABEL=GLOB of metrics.csv files, e.g. "EBT=runs/logs/csv/ebt-xxs-scaling-105k/version_*/metrics.csv"')
    parser.add_argument("--benchmark", action="append", default=[], help="LABEL=PATH to benchmark_<model>.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    curves = {}
    for label, pattern in _pairs(args.run):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no metrics.csv matches {pattern}")
        curves[label] = read_valid_rows(paths)
    benchmarks = {label: json.load(open(path)) for label, path in _pairs(args.benchmark)}
    md = build_result_md(curves, benchmarks)
    with open(args.out, "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
