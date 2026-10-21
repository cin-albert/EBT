"""Shared evaluation harness for the thinking study: checkpoints, eval data, and the EBT MCMC loop.

The MCMC loop re-implements `EBT_NLP.forward` (model/nlp/ebt.py) for inference so that every
intermediate quantity can be recorded. Semantics, matched to the repo code:

- state z_t (B, S, V) is treated as logits; z_0 ~ N(0, 1) (`denoising_initial_condition random_noise`)
- each step: p_t = softmax(z_t + Langevin noise); E_t = E(p_t); g_t = dE/dp_t  (gradient w.r.t. the
  *probabilities*, not the logits); z_{t+1} = p_t - alpha * g_t
- the prediction after the last step is softmax(z_N); the loss is CE(log_softmax(z_N), target)
- landscape (time embedding) index of step t is min(t, mcmc_num_steps - 1)
"""
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, random_split

REPO = Path(__file__).resolve().parents[3]
REPRO_CKPTS = REPO / "experiments/reproduce_xxs_scaling/runs/logs/checkpoints"
SCALE_CKPTS = REPO / "experiments/scale_to_16m_sequences/runs/logs/checkpoints"
RUN_DIRS = {  # key -> (checkpoint root, run-directory glob); trainer dirs are f"{run_name}_{timestamp}_"
    "tpp": (REPRO_CKPTS, "baseline-xxs-scaling-105k_2026-10-13_07-23-18_"),
    "s1": (REPRO_CKPTS, "ebt-xxs-scaling-105k_2026-10-13_09-09-28_"),
    "s2": (REPRO_CKPTS, "ebt-s2-xxs-scaling-105k_2026-10-14_05-20-41_"),
    "tpp_b64_16m": (SCALE_CKPTS, "tpp-b64-16m_20*"),
    "s2_b64_16m": (SCALE_CKPTS, "s2-b64-16m_20*"),
}
SPLIT_SEED = 33  # seed of the reproduction runs; DDP reset_seed re-seeds with it right before random_split
PAD_ID = 0  # GPT-NeoX eos == pad; the training loss ignores this target id
SEQ_LEN = 256


def ckpt_path(model: str, step: int = 105000) -> Path:
    root, pattern = RUN_DIRS[model]
    matches = sorted(p for d in root.glob(pattern) for p in d.glob(f"epoch=epoch=*-step=step={step}-*.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one checkpoint for {model} step {step}, found {matches}")
    return matches[0]


def load_model(model: str, step: int = 105000, device: str = "cuda"):
    """Return (inner model, hparams dict) in eval mode; inner model is EBT_NLP or Baseline_Transformer_NLP."""
    from base_model_trainer import ModelTrainer

    ckpt = torch.load(ckpt_path(model, step), map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    trainer = ModelTrainer(hparams)
    trainer.load_state_dict(ckpt["state_dict"])
    torch.set_float32_matmul_precision(hparams["set_matmul_precision"])  # match training numerics
    return trainer.model.to(device).eval(), hparams


def validation_split(hparams: dict) -> Subset:
    """Validation set of a run: the fixed indices if the run used them, else the reproduction's seeded split."""
    from data.fixed_val_split import fixed_val_split
    from data.nlp.pajama_subset import RedPajamaSubsetDataset

    if hparams.get("fixed_val_indices"):
        return fixed_val_split(RedPajamaSubsetDataset(Namespace(**hparams)), hparams["fixed_val_indices"])[1]
    L.seed_everything(SPLIT_SEED, verbose=False)
    full = RedPajamaSubsetDataset(Namespace(**hparams))
    n_train = int(len(full) * (1 - hparams["validation_split_pct"]))
    return random_split(full, [n_train, len(full) - n_train])[1]


def eval_subset_indices(n_val: int, n: int, seed: int = 0) -> list[int]:
    """Fixed random subset of validation positions used by all sweeps."""
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_val, generator=g)[:n].tolist()


def collate_fixed(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad to SEQ_LEN + 1 and return (inputs, targets, mask), each (B, SEQ_LEN)."""
    ids = torch.full((len(items), SEQ_LEN + 1), PAD_ID, dtype=torch.long)
    for i, item in enumerate(items):
        seq = torch.tensor(item["input_ids"][: SEQ_LEN + 1])
        ids[i, : len(seq)] = seq
    targets = ids[:, 1:]
    return ids[:, :-1], targets, targets != PAD_ID


def batches(ds, indices: list[int], batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield collate_fixed([ds[i] for i in indices[start : start + batch_size]])


def eval_dataset(data: str, hparams: dict):
    """RedPajama validation ("rp_val") or one of the paper's OOD sets (ood_data.OOD_SETS)."""
    if data == "rp_val":
        return validation_split(hparams)
    from ood_data import OOD_SETS, OODDataset, ood_texts
    if data not in OOD_SETS:
        raise ValueError(f"unknown eval data {data!r}; expected rp_val or one of {OOD_SETS}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hparams["tokenizer"], clean_up_tokenization_spaces=False)
    return OODDataset(ood_texts(data), tok)


def answer_masks(ds, indices: list[int], batch_size: int):
    """Per batch of `batches`, (B, SEQ_LEN) bool: target j is in the answer iff j + 1 >= n_prompt."""
    for start in range(0, len(indices), batch_size):
        items = [ds[i] for i in indices[start : start + batch_size]]
        m = torch.zeros(len(items), SEQ_LEN, dtype=torch.bool)
        for row, item in enumerate(items):
            if "n_prompt" in item:
                m[row, max(item["n_prompt"] - 1, 0):] = True
        yield m


def landscape_index(model, t: int) -> int:
    return min(t, model.hparams.mcmc_num_steps - 1)


def energy(model, ctx_emb: torch.Tensor, p: torch.Tensor, landscape: int) -> torch.Tensor:
    """Per-token energy (B, S) of candidate distributions p (B, S, V) given context embeddings (B, S, D)."""
    pred_emb = model.vocab_to_embed(p)
    return model.transformer(torch.cat((ctx_emb, pred_emb), dim=1), start_pos=0, mcmc_step=landscape).squeeze(-1)


def energy_from_embedding(model, ctx_emb: torch.Tensor, pred_emb: torch.Tensor, landscape: int) -> torch.Tensor:
    return model.transformer(torch.cat((ctx_emb, pred_emb), dim=1), start_pos=0, mcmc_step=landscape).squeeze(-1)


@dataclass
class Step:
    t: int
    z: torch.Tensor  # state before the step (B, S, V), detached
    p: torch.Tensor  # softmax(z + noise), the point where energy and gradient are taken
    energy: torch.Tensor  # (B, S)
    grad: torch.Tensor  # dE/dp (B, S, V)
    landscape: int


def mcmc(model, input_ids: torch.Tensor, n_steps: int, noise_std: float, alpha: float | None = None,
         z0: torch.Tensor | None = None):
    """Run the EBT inference loop; yields a `Step` per MCMC step, then returns z_N (via StopIteration.value).

    Random draws follow the repo's order (initial noise, then one Langevin draw per step), so with the same
    torch seed this matches `EBT_NLP.forward(..., no_randomness=True)` for the same number of steps.
    """
    alpha = float(model.alpha.clamp(min=1e-4)) if alpha is None else alpha
    with torch.no_grad():
        ctx_emb = model.embeddings(input_ids)
        z = model.corrupt_embeddings(ctx_emb) if z0 is None else z0
    for t in range(n_steps):
        z = z.detach()
        noisy = z + torch.randn_like(z) * noise_std if noise_std > 0 else z
        p = F.softmax(noisy, dim=-1).requires_grad_()
        lsc = landscape_index(model, t)
        with torch.enable_grad():
            e = energy(model, ctx_emb, p, lsc)
            (g,) = torch.autograd.grad(e.sum(), p)
        yield Step(t, z, p.detach(), e.detach(), g, lsc)
        z = p.detach() - alpha * g
    return z.detach()


def run_mcmc(model, input_ids: torch.Tensor, n_steps: int, noise_std: float, alpha: float | None = None,
             on_step=None) -> torch.Tensor:
    """Drive `mcmc` to the end, calling on_step(step) for each step; returns final logits z_N."""
    gen = mcmc(model, input_ids, n_steps, noise_std, alpha)
    while True:
        try:
            step = next(gen)
        except StopIteration as stop:
            return stop.value
        if on_step is not None:
            on_step(step)


def token_logprob(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target) under softmax(logits); (B, S)."""
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(1e-30))).sum(-1)


def tpp_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids, return_raw_logits=True)


def train_eval_steps(hparams: dict) -> int:
    """Number of MCMC steps the repo's validation uses (max of the randomized range for S2)."""
    if hparams.get("randomize_mcmc_num_steps", 0) > 0:
        return (hparams["mcmc_num_steps"] - 1) + hparams["randomize_mcmc_num_steps"] + 1
    return hparams["mcmc_num_steps"]
