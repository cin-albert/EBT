"""Per-step arrays of one step-size arm on one eval set; same fields, sequences and seeds as the momentum grid.

Writes results/single/<arm>_<set>/{part_a_noise0,part_a_noise3}.npz and config.json (written last; run_all.sh
skips a run that has it).
Usage: python run_decay.py --arm gd_a5_harm --data gsm8k --out-dir results/single/gd_a5_harm_gsm8k
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

from decay_arms import DECAY_ARMS
from harness import eval_dataset, eval_subset_indices, load_model
from run_momentum import record, save_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(DECAY_ARMS), required=True)
    ap.add_argument("--data", required=True, help="rp_val or an OOD set (ood_data.OOD_SETS)")
    ap.add_argument("--n-seqs", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arm = DECAY_ARMS[args.arm]
    model, hparams = load_model("s2_paper", 1_000_000)
    ds = eval_dataset(args.data, hparams)
    idx = eval_subset_indices(len(ds), args.n_seqs)
    t0 = time.time()
    for noise in sorted({0.0, float(hparams["langevin_dynamics_noise"])}):
        save_record(args.out_dir / f"part_a_noise{noise:g}.npz",
                    record(model, ds, idx, args.n_steps, noise, arm, args.batch_size, args.seed))
    config = vars(args) | {"out_dir": str(args.out_dir), "arm_spec": dataclasses.asdict(arm),
                           "n_seqs_used": len(idx), "seconds": round(time.time() - t0, 1)}
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(config, indent=1))


if __name__ == "__main__":
    main()
