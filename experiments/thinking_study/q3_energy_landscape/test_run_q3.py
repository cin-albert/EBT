import numpy as np
import pytest
import torch

import run_q3
from fake_ebt import FakeEBT
from harness import mcmc, token_logprob
from model.nlp.mcmc_update import RULES, next_logits


@pytest.mark.parametrize("rule", RULES)
def test_line_profile_at_s1_is_the_rules_actual_update(rule):
    model = FakeEBT(rule)
    x = torch.tensor([[1, 2, 3]])
    targets, mask = torch.tensor([[4, 5, 6]]), torch.ones(1, 3, dtype=torch.bool)
    torch.manual_seed(0)
    s = next(mcmc(model, x, 2, 0.0))
    ctx = model.embeddings(x).detach()
    line = run_q3.line_profile(model, ctx, s, float(model.alpha), targets, mask, rule)
    i = int(np.argmin(np.abs(run_q3.LINE_S - 1.0)))
    expected = token_logprob(next_logits(rule, s.z, s.p, s.grad, float(model.alpha)), targets)[mask]
    assert torch.allclose(line["logp"][i], expected, atol=1e-6)
