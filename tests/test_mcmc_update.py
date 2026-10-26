import pytest
import torch

from model.nlp.mcmc_update import RULES, descent_direction, next_logits

V = 11


def _point(seed=0, shape=(2, 3)):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(*shape, V, generator=g, dtype=torch.float64)
    w = torch.randn(V, 4, generator=g, dtype=torch.float64)
    return z, w


def _energy(p, w):
    return ((p @ w) ** 2).sum(-1)  # any smooth E(p) works


def test_probs_rule_is_the_repo_update():
    z, w = _point()
    p = torch.softmax(z, -1).requires_grad_()
    (g,) = torch.autograd.grad(_energy(p, w).sum(), p)
    assert torch.equal(next_logits("probs", z, p, g, 5.0), p - 5.0 * g)


def test_logits_rule_matches_autograd_through_softmax():
    z, w = _point()
    z = z.requires_grad_()
    p = torch.softmax(z, -1)
    g_p, g_z = torch.autograd.grad(_energy(p, w).sum(), [p, z])
    out = next_logits("logits", z.detach(), p.detach(), g_p, 0.7)
    assert torch.allclose(out, z.detach() - 0.7 * g_z, atol=1e-12)


def test_log_probs_rule_keeps_p_where_the_gradient_is_constant():
    z, _ = _point()
    p = torch.softmax(z, -1)
    g = torch.full_like(p, 2.5)
    assert torch.allclose(torch.softmax(next_logits("log_probs", z, p, g, 5.0), -1), p, atol=1e-12)


def test_probs_rule_moves_off_a_non_uniform_point_even_with_constant_gradient():
    z, _ = _point()
    p = torch.softmax(z, -1)
    g = torch.full_like(p, 2.5)
    assert not torch.allclose(torch.softmax(next_logits("probs", z, p, g, 5.0), -1), p, atol=1e-3)


@pytest.mark.parametrize("rule", RULES)
def test_descent_direction_vanishes_for_a_constant_gradient(rule):
    z, _ = _point()
    p = torch.softmax(z, -1)
    assert torch.allclose(descent_direction(rule, p, torch.full_like(p, -1.3)), torch.zeros_like(p), atol=1e-12)


def test_descent_direction_of_logits_is_the_logit_gradient():
    z, w = _point()
    z = z.requires_grad_()
    p = torch.softmax(z, -1)
    g_p, g_z = torch.autograd.grad(_energy(p, w).sum(), [p, z])
    assert torch.allclose(descent_direction("logits", p.detach(), g_p), g_z, atol=1e-12)


@pytest.mark.parametrize("rule", RULES)
def test_next_logits_broadcasts_per_token_alpha(rule):
    z, w = _point()
    p = torch.softmax(z, -1).requires_grad_()
    (g,) = torch.autograd.grad(_energy(p, w).sum(), p)
    alpha = torch.rand(2, 3, 1, dtype=torch.float64) + 0.5
    out = next_logits(rule, z, p.detach(), g, alpha)
    one = next_logits(rule, z[0, 1], p.detach()[0, 1], g[0, 1], float(alpha[0, 1]))
    assert out.shape == z.shape and torch.allclose(out[0, 1], one, atol=1e-12)


def test_unknown_rule_raises():
    z, _ = _point()
    p = torch.softmax(z, -1)
    with pytest.raises(ValueError):
        next_logits("energy", z, p, p, 1.0)
    with pytest.raises(ValueError):
        descent_direction("energy", p, p)
