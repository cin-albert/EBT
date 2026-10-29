from pathlib import Path

import numpy as np
import pytest
import torch

WS = Path(__file__).resolve().parent
GD_A5 = WS.parent / "energy_momentum/results/single/gd_a5_rp_val/part_a_noise3.npz"


@pytest.mark.skipif(not torch.cuda.is_available() or not GD_A5.exists(), reason="needs a GPU and the momentum grid")
def test_const_schedule_reproduces_gd_a5_grid():
    import harness
    from momentum import Arm
    from run_momentum import record

    model, hp = harness.load_model("s2_paper", 1_000_000)
    ds = harness.eval_dataset("rp_val", hp)
    idx = harness.eval_subset_indices(len(ds), 1024)[:64]
    a = record(model, ds, idx, 32, 3.0, Arm("x", "gd", 0.0, 5.0, "const"), batch_size=16, seed=0)
    ref = np.load(GD_A5)
    n = a["logp"].shape[1]
    assert np.array_equal(a["seq_id"], ref["seq_id"][:n])
    assert np.array_equal(a["logp"], ref["logp"][:, :n]) and np.array_equal(a["logp_final"], ref["logp_final"][:n])
