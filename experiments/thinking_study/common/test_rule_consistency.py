"""EBT_NLP.forward and harness.mcmc must apply the same update rule (S2 105k checkpoint on CPU)."""
import pytest
import torch

import harness
from model.nlp.mcmc_update import RULES


@pytest.fixture(scope="module")
def s2():
    try:
        harness.ckpt_path("s2")
    except FileNotFoundError:
        pytest.skip("S2 105k checkpoint not available")
    model, _ = harness.load_model("s2", device="cpu")
    return model


@pytest.mark.parametrize("rule", RULES)
def test_forward_and_harness_agree(s2, rule):
    s2.hparams.mcmc_update_rule = rule
    x = torch.randint(1, 1000, (2, 16), generator=torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    dists, _ = s2(x, learning=False, no_randomness=True)
    torch.manual_seed(0)
    z = harness.run_mcmc(s2, x, len(dists), float(s2.langevin_dynamics_noise_std))
    assert torch.allclose(dists[-1].reshape(2, 16, -1), torch.log_softmax(z, -1), atol=1e-4)


def test_logits_rule_backpropagates_to_the_model(s2):
    s2.hparams.mcmc_update_rule = "logits"
    s2.zero_grad()
    x = torch.randint(1, 1000, (1, 8), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    dists, _ = s2(x, learning=True, no_randomness=True)
    dists[-1].mean().backward()
    grads = [p.grad for p in s2.transformer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)


def test_advanced_inference_rejects_other_rules(s2):
    s2.hparams.mcmc_update_rule = "log_probs"
    with pytest.raises(NotImplementedError):
        s2._run_ebt_inference_steps(None, None, None, None, 0, False)
