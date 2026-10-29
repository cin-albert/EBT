from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import harness
from momentum import ARMS, Arm, momentum_mcmc, run_momentum_mcmc, step_size

V, D, B, S = 7, 4, 2, 3


def stub_model(nonlinear: bool = True, seed: int = 0):
    """Minimal stand-in for EBT_NLP: energy = sum over D of (tanh)(W_v p) * w, per predicted position."""
    torch.manual_seed(seed)
    w = torch.randn(D)

    def transformer(x, start_pos=0, mcmc_step=0):
        h = x[:, x.shape[1] // 2:]
        h = torch.tanh(h) if nonlinear else h
        return (h @ w).unsqueeze(-1)

    return SimpleNamespace(
        embeddings=nn.Embedding(11, D),
        vocab_to_embed=nn.Linear(V, D),
        transformer=transformer,
        corrupt_embeddings=lambda e: torch.randn(e.shape[0], e.shape[1], V),
        alpha=torch.tensor(5.0),
        hparams=SimpleNamespace(mcmc_num_steps=1),
    )


def ids():
    return torch.tensor([[1, 2, 3], [4, 5, 6]])


def collect(gen):
    steps = []
    while True:
        try:
            steps.append(next(gen))
        except StopIteration as stop:
            return steps, stop.value


def run(model, arm, n_steps=6, noise=3.0, seed=1):
    torch.manual_seed(seed)
    return collect(momentum_mcmc(model, ids(), n_steps, noise, arm))


def test_gd_matches_harness_mcmc():
    model = stub_model()
    torch.manual_seed(1)
    ref_steps, ref_z = collect(harness.mcmc(model, ids(), 6, 3.0, alpha=5.0))
    steps, z = run(model, Arm("gd_a5", "gd", 0.0, 5.0))
    assert torch.equal(z, ref_z)
    for a, b in zip(steps, ref_steps):
        assert torch.equal(a.energy, b.energy) and torch.equal(a.grad, b.grad) and torch.equal(a.p, b.p)


def test_hb_and_ema_with_mu0_equal_gd():
    model = stub_model()
    _, z_gd = run(model, Arm("g", "gd"))
    _, z_hb = run(model, Arm("h", "hb", 0.0))
    _, z_ema = run(model, Arm("e", "ema", 0.0))
    assert torch.equal(z_hb, z_gd) and torch.equal(z_ema, z_gd)


def test_ema_equals_gd_for_constant_gradient():
    model = stub_model(nonlinear=False)
    _, z_gd = run(model, Arm("g", "gd"))
    _, z_ema = run(model, Arm("e", "ema", 0.7))
    assert torch.allclose(z_ema, z_gd, atol=1e-6)


def test_hb_constant_gradient_step_sizes():
    model, mu, alpha = stub_model(nonlinear=False), 0.7, 5.0
    steps, z_final = run(model, Arm("h", "hb", mu, alpha), n_steps=4, noise=0.0)
    nexts = [s.z for s in steps[1:]] + [z_final]
    for t, (s, z_next) in enumerate(zip(steps, nexts)):
        expected = -alpha * s.grad * (1 - mu ** (t + 1)) / (1 - mu)
        assert torch.allclose(z_next - s.p, expected, atol=1e-5)


def test_ema_first_step_equals_gd():
    model = stub_model()
    gd_steps, _ = run(model, Arm("g", "gd"), n_steps=2)
    ema_steps, _ = run(model, Arm("e", "ema", 0.9), n_steps=2)
    assert torch.equal(ema_steps[1].z, gd_steps[1].z)


def test_velocity_resets_per_call():
    model = stub_model()
    arm = Arm("h", "hb", 0.9)
    _, z1 = run(model, arm)
    _, z2 = run(model, arm)
    assert torch.equal(z1, z2)


def test_run_momentum_mcmc_returns_final_state_and_calls_back():
    model, seen = stub_model(), []
    arm = Arm("e", "ema", 0.5)
    torch.manual_seed(1)
    z = run_momentum_mcmc(model, ids(), 3, 3.0, arm, on_step=lambda s: seen.append(s.t))
    _, z_ref = run(model, arm, n_steps=3)
    assert seen == [0, 1, 2] and torch.equal(z, z_ref)


def test_arms_registry():
    assert len(ARMS) == 13
    assert ARMS["gd_a5"] == Arm("gd_a5", "gd", 0.0, 5.0)
    assert {a.alpha for n, a in ARMS.items() if n.startswith("gd_")} == {5.0, 7.5, 10.0, 15.0, 20.0}
    assert sorted(a.mu for a in ARMS.values() if a.rule == "ema") == [0.3, 0.5, 0.7, 0.9]
    assert all(a.alpha == 5.0 for a in ARMS.values() if a.rule != "gd")
    assert all(name == a.name for name, a in ARMS.items())


def test_step_size_schedules():
    assert [step_size(Arm("c", "gd"), t) for t in range(3)] == [5.0, 5.0, 5.0]
    harm = [step_size(Arm("h", "gd", 0.0, 5.0, "harmonic"), t) for t in range(3)]
    assert harm[:2] == [5.0, 2.5] and abs(harm[2] - 5 / 3) < 1e-12
    assert [step_size(Arm("g", "gd", 0.0, 5.0, "geom", 0.5), t) for t in range(3)] == [5.0, 2.5, 1.25]


def test_decay_scales_each_update():
    model = stub_model(nonlinear=False)
    steps, z_final = run(model, Arm("h", "gd", 0.0, 5.0, "harmonic"), n_steps=4, noise=0.0)
    nexts = [s.z for s in steps[1:]] + [z_final]
    for t, (s, z_next) in enumerate(zip(steps, nexts)):
        assert torch.allclose(z_next - s.p, -5.0 / (t + 1) * s.grad, atol=1e-6)


def test_unknown_schedule_rejected():
    import pytest
    with pytest.raises(ValueError):
        Arm("x", "gd", 0.0, 5.0, "harmonc")
