import os
from pathlib import Path

import numpy as np
import pytest
import torch

from momentum import ARMS, Arm
from run_momentum import FIELDS, record, save_record
from test_momentum import stub_model

WS = Path(__file__).resolve().parent
Q3_RP_VAL = WS.parent / "results/q3/s2_paper_step1000000_rp_val/part_a_noise3.npz"


def toy_dataset(n_seqs: int = 5):
    g = torch.Generator().manual_seed(0)  # token ids < 7 (stub vocab); 0 is the pad id
    return [{"input_ids": torch.randint(1, 7, (int(torch.randint(4, 9, (1,), generator=g)),), generator=g).tolist()}
            for _ in range(n_seqs)]


def stub_record(n_steps=12, noise=3.0, arm=Arm("e", "ema", 0.5)):
    ds = toy_dataset()
    return record(stub_model(), ds, list(range(len(ds))), n_steps, noise, arm, batch_size=2, seed=0, device="cpu")


def test_record_fields_match_q3(tmp_path):
    a = stub_record()
    n_tok = a["logp_final"].shape[0]
    assert set(FIELDS) | {"logp_final", "seq_id"} <= set(a)
    for k in FIELDS:
        assert a[k].shape == (12, n_tok), k
    assert np.all(np.diff(a["seq_id"]) >= 0) and a["seq_id"].max() == 4
    save_record(tmp_path / "part_a_noise3.npz", a)
    assert set(np.load(tmp_path / "part_a_noise3.npz").files) == set(a)


def test_cycle_metrics_accepts_output():
    from analyze_q3_extra import cycle_metrics

    share = cycle_metrics(stub_record())["share_two_cycle"]
    assert 0.0 <= share <= 1.0


def test_record_is_seeded_per_batch():
    a, b = stub_record(), stub_record()
    assert np.array_equal(a["logp"], b["logp"])


@pytest.mark.skipif(not torch.cuda.is_available() or not Q3_RP_VAL.exists(), reason="needs GPU and Q3 outputs")
def test_gd_a5_reproduces_q3_part_a(monkeypatch):
    import harness

    monkeypatch.setenv("EBT_VAL_SUBSET", os.environ.get("EBT_VAL_SUBSET", str(WS.parent / "data/val_subset")))
    model, hparams = harness.load_model("s2_paper", 1000000)
    ds = harness.eval_dataset("rp_val", hparams)
    idx = harness.eval_subset_indices(len(ds), 1024)[:64]
    a = record(model, ds, idx, 32, 3.0, ARMS["gd_a5"], batch_size=16, seed=0)
    ref = np.load(Q3_RP_VAL)["logp"][:, : a["logp"].shape[1]]
    np.testing.assert_allclose(a["logp"], ref, atol=1e-5)
