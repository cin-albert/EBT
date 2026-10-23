import numpy as np

import harness


def test_ckpt_path_finds_any_epoch_and_new_run_keys(tmp_path, monkeypatch):
    run = tmp_path / "s2-b64-16m_2026-10-17_00-00-00_"
    run.mkdir()
    (run / "epoch=epoch=1-step=step=250000-valid_loss=valid_loss=3.9000.ckpt").touch()
    monkeypatch.setitem(harness.RUN_DIRS, "s2_b64_16m", (tmp_path, "s2-b64-16m_20*"))
    assert harness.ckpt_path("s2_b64_16m", 250000).name.startswith("epoch=epoch=1-step=step=250000")


def test_run_glob_does_not_match_longer_run_name(tmp_path, monkeypatch):
    for name in ("s2-b64-16m_2026-10-17_00-00-00_", "s2-b64-16m-seed2_2026-10-17_00-00-00_"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "epoch=epoch=0-step=step=25000-valid_loss=valid_loss=4.0.ckpt").touch()
    monkeypatch.setitem(harness.RUN_DIRS, "s2_b64_16m", (tmp_path, "s2-b64-16m_20*"))
    assert "seed2" not in str(harness.ckpt_path("s2_b64_16m", 25000))


def test_validation_split_uses_fixed_indices_when_present(tmp_path, monkeypatch):
    np.save(tmp_path / "val.npy", np.array([4, 0, 2], dtype=np.int64))
    monkeypatch.setattr("data.nlp.pajama_subset.RedPajamaSubsetDataset", lambda hp: list(range(6)))
    val = harness.validation_split({"fixed_val_indices": str(tmp_path / "val.npy"), "validation_split_pct": 0.5})
    assert [val[i] for i in range(len(val))] == [4, 0, 2]


def test_validation_split_falls_back_to_seeded_random_split(monkeypatch):
    monkeypatch.setattr("data.nlp.pajama_subset.RedPajamaSubsetDataset", lambda hp: list(range(100)))
    a = harness.validation_split({"validation_split_pct": 0.1})
    b = harness.validation_split({"validation_split_pct": 0.1})
    assert len(a) == 10 and list(a.indices) == list(b.indices)


def test_paper_run_keys_resolve(tmp_path, monkeypatch):
    assert harness.PAPER_CKPTS == harness.REPO / "experiments/scale_to_paper_level/runs/logs/checkpoints"
    for key, run in (("s2_paper", "s2-paper"), ("tpp_paper", "tpp-paper")):
        root, pattern = harness.RUN_DIRS[key]
        assert root == harness.PAPER_CKPTS
        d = tmp_path / f"{run}_2026-11-01_00-00-00_"
        d.mkdir()
        ckpt = d / "epoch=epoch=1-step=step=1000000-valid_loss=valid_loss=3.5.ckpt"
        ckpt.touch()
        monkeypatch.setitem(harness.RUN_DIRS, key, (tmp_path, pattern))
        assert harness.ckpt_path(key, 1000000) == ckpt
