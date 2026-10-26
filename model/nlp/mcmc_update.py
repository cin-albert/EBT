"""Update rules of the EBT NLP MCMC loop: next logits from the step's (noisy) logits z, p = softmax(z) and g = dE/dp.

- probs (repo default): z' = p - alpha * g. Fixed points are the stationary points on the simplex of
  alpha * E(p) - H(p) - 0.5 * ||p||^2 (docs/EXPLAIN_FIXED_POINT.md), not of E.
- logits (paper pseudocode): z' = z - alpha * dE/dz, with dE/dz = p * (g - <p, g>) by the chain rule through softmax.
- log_probs (exponentiated gradient, mirror descent on the simplex): z' = log p - alpha * g. Fixed points are the
  stationary points of E on the simplex.
"""
import torch

RULES = ("probs", "logits", "log_probs")


def _check(rule: str) -> None:
    if rule not in RULES:
        raise ValueError(f"unknown MCMC update rule {rule!r}; expected one of {RULES}")


def descent_direction(rule: str, p: torch.Tensor, grad_p: torch.Tensor) -> torch.Tensor:
    """Direction the rule moves along, without the component softmax ignores; zero exactly at its fixed points."""
    _check(rule)
    if rule == "logits":
        return p * (grad_p - (p * grad_p).sum(-1, keepdim=True))
    return grad_p - grad_p.mean(-1, keepdim=True)


def next_logits(rule: str, z: torch.Tensor, p: torch.Tensor, grad_p: torch.Tensor, alpha) -> torch.Tensor:
    _check(rule)
    if rule == "probs":
        return p - alpha * grad_p
    if rule == "log_probs":
        return torch.log_softmax(z, dim=-1) - alpha * grad_p
    return z - alpha * descent_direction(rule, p, grad_p)
