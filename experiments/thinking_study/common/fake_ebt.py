"""Tiny CPU stand-in for EBT_NLP with the attributes harness.mcmc and Q3 use (test utility)."""
import torch
from lightning_fabric.utilities.data import AttributeDict
from torch import nn

V, D = 9, 4


class _Energy(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(D, D, generator=torch.Generator().manual_seed(1)))

    def forward(self, x, start_pos=0, mcmc_step=0):  # (B, 2S, D) -> per-token energy of the predicted half, (B, S, 1)
        pred = x[:, x.shape[1] // 2:]
        return ((pred @ self.w) ** 2).sum(-1, keepdim=True) * (1 + mcmc_step)


class FakeEBT(nn.Module):
    def __init__(self, rule=None, n_steps=2):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.embeddings = nn.Embedding(V, D)
        self.vocab_to_embed = nn.Linear(V, D, bias=False)
        with torch.no_grad():
            self.embeddings.weight.copy_(torch.randn(V, D, generator=g))
            self.vocab_to_embed.weight.copy_(torch.randn(D, V, generator=g))
        self.transformer = _Energy()
        self.alpha = nn.Parameter(torch.tensor(0.8), requires_grad=False)
        hp = {"mcmc_num_steps": n_steps}
        if rule is not None:
            hp["mcmc_update_rule"] = rule
        self.hparams = AttributeDict(hp)

    def corrupt_embeddings(self, emb):
        return torch.randn(*emb.shape[:2], V)
