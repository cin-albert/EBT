"""EBT inference loop with momentum on the energy gradient (IMPLEMENTATION_PLAN.md §0.1).

The repo loop re-anchors at the probabilities every step: p_t = softmax(z_t + noise), g_t = dE/dp at p_t,
z_{t+1} = p_t - alpha * u_t. The rules differ only in u_t:
  gd : u_t = g_t                                                  (bit-identical to harness.mcmc)
  hb : v_t = mu v_{t-1} + g_t;               u_t = v_t           (heavy-ball; steady-state step alpha / (1 - mu))
  ema: m_t = mu m_{t-1} + (1 - mu) g_t;      u_t = m_t / (1 - mu^(t+1))   (bias-corrected; step-size neutral)
Random draws follow harness.mcmc's order, so with the same torch seed every arm sees the same noise.
"""
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from harness import Step, energy, landscape_index


@dataclass(frozen=True)
class Arm:
    name: str
    rule: Literal["gd", "hb", "ema"]
    mu: float = 0.0
    alpha: float = 5.0


ARMS: dict[str, Arm] = {
    a.name: a
    for a in [Arm(f"gd_a{al:g}", "gd", 0.0, al) for al in (5.0, 7.5, 10.0, 15.0, 20.0)]
    + [Arm(f"{rule}_m{mu:g}", rule, mu) for rule in ("hb", "ema") for mu in (0.3, 0.5, 0.7, 0.9)]
}


def momentum_mcmc(model, input_ids: torch.Tensor, n_steps: int, noise_std: float, arm: Arm,
                  z0: torch.Tensor | None = None):
    """Same contract as harness.mcmc: yields a Step per MCMC step (before its update), returns z_N."""
    with torch.no_grad():
        ctx_emb = model.embeddings(input_ids)
        z = model.corrupt_embeddings(ctx_emb) if z0 is None else z0
    v = None
    for t in range(n_steps):
        z = z.detach()
        noisy = z + torch.randn_like(z) * noise_std if noise_std > 0 else z
        p = F.softmax(noisy, dim=-1).requires_grad_()
        lsc = landscape_index(model, t)
        with torch.enable_grad():
            e = energy(model, ctx_emb, p, lsc)
            (g,) = torch.autograd.grad(e.sum(), p)
        yield Step(t, z, p.detach(), e.detach(), g, lsc)
        v = g if v is None else arm.mu * v + g  # heavy-ball sum s_t
        if arm.rule == "gd":
            u = g
        elif arm.rule == "hb":
            u = v
        else:  # m_t / (1 - mu^(t+1)) == s_t / sum_{i<=t} mu^i; this form is exact at t = 0
            u = v / sum(arm.mu ** i for i in range(t + 1))
        z = p.detach() - arm.alpha * u
    return z.detach()


def run_momentum_mcmc(model, input_ids: torch.Tensor, n_steps: int, noise_std: float, arm: Arm,
                      on_step=None) -> torch.Tensor:
    """Drive `momentum_mcmc` to the end, calling on_step(step) for each step; returns final logits z_N."""
    gen = momentum_mcmc(model, input_ids, n_steps, noise_std, arm)
    while True:
        try:
            step = next(gen)
        except StopIteration as stop:
            return stop.value
        if on_step is not None:
            on_step(step)
