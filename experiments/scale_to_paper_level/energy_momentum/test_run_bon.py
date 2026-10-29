import torch
import torch.nn.functional as F

import harness
from momentum import Arm, run_momentum_mcmc
from run_bon import record_bon
from test_momentum import stub_model
from test_run_momentum import toy_dataset

ARM = Arm("e", "ema", 0.5)


def bon(model, ds):
    return record_bon(model, ds, list(range(len(ds))), ARM, k=3, n_candidates=3, noise=3.0, batch_size=2, seed=0,
                      device="cpu")


def test_bon_shapes_and_seeds():
    model, ds = stub_model(), toy_dataset()
    a = bon(model, ds)
    n_tok = a["seq_id"].shape[0]
    assert a["e_final"].shape == (3, n_tok) and a["logp"].shape == (3, n_tok)
    inputs, targets, mask = next(harness.batches(ds, [0, 1], 2))
    torch.manual_seed(0 * 1_000_003 + 0 * 1009 + 0)
    z = run_momentum_mcmc(model, inputs, 3, 3.0, ARM)
    ref = harness.token_logprob(z, targets)[mask].numpy()
    assert (a["logp"][0, : len(ref)] == ref).all()
    assert not (a["logp"][0] == a["logp"][1]).all()  # candidates differ


def test_energy_on_last_landscape():
    model, ds = stub_model(), toy_dataset()
    a = bon(model, ds)
    inputs, targets, mask = next(harness.batches(ds, [0, 1], 2))
    torch.manual_seed(0)
    z = run_momentum_mcmc(model, inputs, 3, 3.0, ARM)
    with torch.no_grad():
        e = harness.energy(model, model.embeddings(inputs), F.softmax(z, -1), model.hparams.mcmc_num_steps - 1)
    assert torch.allclose(torch.from_numpy(a["e_final"][0, : int(mask.sum())]), e[mask])
