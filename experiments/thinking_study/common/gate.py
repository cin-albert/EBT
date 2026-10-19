"""M0 correctness gate: the harness must reproduce the logged validation numbers before any result is used.

Checks
1. Each checkpoint, run through the repo's own `forward_loss_wrapper` on the rebuilt validation split,
   reproduces the losses logged at step 105k (validates checkpoint loading and the split).
2. `harness.mcmc` gives the same final logits as `EBT_NLP.forward` under the same seed (validates the loop).
3. Energies are block-diagonal across positions: perturbing the prediction at one position leaves the
   energies of all other positions unchanged (the per-token independence that Q2/Q3 rely on).

Usage: python common/gate.py --out common/results/gate.json
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader

from data.nlp.collator import NLP_HF_Collator
from harness import (batches, energy, eval_subset_indices, load_model, mcmc, run_mcmc, train_eval_steps,
                     validation_split)

LOGGED = {  # from ../reproduce_xxs_scaling/runs/logs/csv/*/version_0/metrics.csv at step 104999
    "tpp": {"loss": 3.6964},
    "s1": {"final_step_loss": 3.8373, "initial_loss": 3.8563},
    "s2": {"final_step_loss": 3.9431, "initial_loss": 6.1643},
}


def logged_style_validation(model, hparams, val_ds) -> dict:
    """Batch-size-weighted mean of per-batch metrics, as Lightning's on_epoch logging does."""
    loader = DataLoader(val_ds, batch_size=hparams["batch_size_per_device"], shuffle=False, num_workers=8,
                        collate_fn=NLP_HF_Collator(Namespace(**hparams)))
    sums, n = {}, 0
    for batch in loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        out = model.forward_loss_wrapper(batch, "valid")
        bs = batch["input_ids"].shape[0]
        for k in ("loss", "initial_loss", "final_step_loss"):
            if k in out:
                sums[k] = sums.get(k, 0.0) + float(out[k]) * bs
        n += bs
    return {k: v / n for k, v in sums.items()}


def check_loop_equivalence(model, hparams, val_ds) -> float:
    inputs, _, _ = next(batches(val_ds, eval_subset_indices(len(val_ds), 8), 8))
    inputs = inputs.cuda()
    torch.manual_seed(123)
    ref = model(inputs, return_raw_logits=True, no_randomness=True)[0][-1].detach()
    torch.manual_seed(123)
    ours = run_mcmc(model, inputs, train_eval_steps(hparams), hparams["langevin_dynamics_noise"])
    return float((ref - ours).abs().max())


def check_block_diagonal(model, val_ds) -> dict:
    inputs, _, _ = next(batches(val_ds, eval_subset_indices(len(val_ds), 4), 4))
    inputs = inputs.cuda()
    torch.manual_seed(7)
    step = next(mcmc(model, inputs, 1, 0.0))
    with torch.no_grad():
        ctx = model.embeddings(inputs)
        base = energy(model, ctx, step.p, step.landscape)
        j = 100
        p2 = step.p.clone()
        p2[:, j] = torch.softmax(torch.randn_like(p2[:, j]) * 5, -1)
        moved = energy(model, ctx, p2, step.landscape)
    diff = (moved - base).abs()
    others = torch.cat([diff[:, :j], diff[:, j + 1:]], dim=1)
    return {"changed_position_max_diff": float(diff[:, j].max()), "other_positions_max_diff": float(others.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {}
    val_ds = None
    for name in ("tpp", "s1", "s2"):
        model, hparams = load_model(name)
        val_ds = val_ds or validation_split(hparams)
        L.seed_everything(33, verbose=False)
        entry = {"n_val_docs": len(val_ds), "logged": LOGGED[name], "reproduced": logged_style_validation(model, hparams, val_ds)}
        if name != "tpp":
            entry["loop_max_abs_diff"] = check_loop_equivalence(model, hparams, val_ds)
            entry["block_diagonal"] = check_block_diagonal(model, val_ds)
        report[name] = entry
        print(name, json.dumps(entry, indent=1), flush=True)
        del model
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
