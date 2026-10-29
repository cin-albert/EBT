"""Table 3 perplexities of our step-1M checkpoints under the repo's own test aggregation.

The repo's test step (base_model_trainer.test_step -> forward_loss_wrapper / get_ppl) logs exp(mean CE of the
batch) per batch, and Lightning's epoch value is the batch-size-weighted mean of those per-batch values. Our
reports use exp(mean CE over all tokens). By Jensen the repo's value is never lower, and the gap grows with the
spread of per-batch CE, i.e. with small batches and short, uneven sequences.

S2 runs the repo's test-time inference: the max of the randomized step count (3), the fixed step size and the
training Langevin noise (σ = 3), which `no_randomness` does not switch off. The repo's NLP inference script
(job_scripts/nlp/inference/ebt.sh: --infer_ebt_advanced --infer_ebt_num_steps 2 --infer_langevin_dynamics_noise 1)
is scored as well, plus k = 3 at σ = 1. Every item of each set is scored, in dataset order.

Usage: GPU_ID=0 python table3_protocol.py   (writes results/table3_protocol/{model}_{set}.npz and summary.json)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent
REPO = WS.parents[1]
OUT = WS / "results/table3_protocol"
BATCH_SIZES = (1, 8, 16, 32)  # 8 is the repo's NLP inference scripts' batch_size_per_device
WORLD_SIZES = (1, 4)  # 4 GPUs, as `--gpus -1` on the paper's 4-GPU nodes
RUNS = {  # name -> (model key, MCMC steps, Langevin σ); None = the repo's test path (train_eval_steps, trained σ)
    "tpp_paper": ("tpp_paper", None, None),
    "s2_paper": ("s2_paper", None, None),
    "s2_paper_k2_s1": ("s2_paper", 2, 1.0),
    "s2_paper_k3_s1": ("s2_paper", 3, 1.0),
}


def token_ppl(ce_sum: np.ndarray, n_tok: np.ndarray) -> float:
    """exp(mean CE over all tokens), the reports' convention."""
    return float(np.exp(ce_sum.sum() / n_tok.sum()))


def lightning_epoch_ppl(ce_sum: np.ndarray, n_tok: np.ndarray, batch_size: int, world_size: int = 1) -> float:
    """Batch-size-weighted mean of per-batch exp(mean CE), with DistributedSampler(shuffle=False) sharding."""
    n = len(ce_sum)
    total = -(-n // world_size) * world_size
    order = np.concatenate([np.arange(n), np.arange(total - n)])  # the sampler pads by repeating the head
    num, den = 0.0, 0
    for rank in range(world_size):
        shard = order[rank:total:world_size]
        for start in range(0, len(shard), batch_size):
            b = shard[start : start + batch_size]
            num += len(b) * np.exp(ce_sum[b].sum() / n_tok[b].sum())
            den += len(b)
    return float(num / den)


def score(model_key: str, data: str, n_steps: int | None = None, noise: float | None = None,
          batch_size: int = 32) -> dict:
    """Per-sequence CE sum and token count over every item of `data`, in dataset order."""
    import torch

    from harness import batches, eval_dataset, load_model, run_mcmc, token_logprob, tpp_logits, train_eval_steps

    model, hp = load_model(model_key, 1_000_000)
    ds = eval_dataset(data, hp)
    is_ebt = hp["model_name"] == "ebt"
    ce_sum, n_tok = [], []
    for b, (inputs, targets, mask) in enumerate(batches(ds, list(range(len(ds))), batch_size)):
        inputs, targets, mask = inputs.cuda(), targets.cuda(), mask.cuda()
        if is_ebt:
            torch.manual_seed(b)
            logits = run_mcmc(model, inputs, n_steps or train_eval_steps(hp),
                              float(hp["langevin_dynamics_noise"]) if noise is None else noise)
        else:
            logits = tpp_logits(model, inputs)
        nll = -token_logprob(logits, targets) * mask
        ce_sum.append(nll.sum(1).double().cpu().numpy())
        n_tok.append(mask.sum(1).cpu().numpy())
    return {"ce_sum": np.concatenate(ce_sum), "n_tok": np.concatenate(n_tok)}


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_ID"]
    sys.path[:0] = [str(REPO), str(REPO / "experiments/thinking_study/common")]
    from ood_data import OOD_SETS

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for run, (model_key, n_steps, noise) in RUNS.items():
        for data in OOD_SETS:
            path = OUT / f"{run}_{data}.npz"
            if not path.exists():
                np.savez(path, **score(model_key, data, n_steps, noise))
            r = np.load(path)
            row = {"n_seqs": int(len(r["n_tok"])), "n_tok": int(r["n_tok"].sum()),
                   "token_ppl": token_ppl(r["ce_sum"], r["n_tok"])}
            for w in WORLD_SIZES:
                for bs in BATCH_SIZES:
                    row[f"epoch_ppl_b{bs}_w{w}"] = lightning_epoch_ppl(r["ce_sum"], r["n_tok"], bs, w)
            summary[f"{run}/{data}"] = row
            print(run, data, {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
