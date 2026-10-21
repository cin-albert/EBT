import importlib.util
import json
from pathlib import Path

import numpy as np

TS = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_q2(d: Path, answer):
    rng = np.random.default_rng(0)
    G, T = 4, len(answer)
    off = ["tpp_T0.5", "tpp_T1.0", "tpp_T2.0", "onehot_target", "onehot_tpp_argmax", "uniform"]
    np.savez(d / "arrays.npz", e_final=rng.normal(size=(G, T)), e_last_logged=rng.normal(size=(G, T)),
             logp=-rng.random((G, T)) * 5, top1=rng.random((G, T)) > 0.5, entropy=rng.random((G, T)) * 3,
             js=rng.random((3, T)), off_e=rng.normal(size=(6, T)), off_logp=-rng.random((6, T)) * 5,
             target=np.zeros(T, int), tpp_logp=-rng.random(T) * 5, answer=np.array(answer))
    (d / "config.json").write_text(json.dumps({"off_trajectory": off}))


def fake_q3(d: Path, answer, with_part_b=False):
    rng = np.random.default_rng(0)
    S, T = 6, len(answer)
    for noise in (0, 3):
        np.savez(d / f"part_a_noise{noise}.npz", logp=-rng.random((S, T)) * 5, logp_final=-rng.random(T) * 5,
                 energy=rng.normal(size=(S, T)), phi=rng.normal(size=(S, T)), entropy=rng.random((S, T)),
                 gnorm=rng.random((S, T)), tv_next=rng.random((S, T)), cos_prev=rng.random((S, T)),
                 answer=np.array(answer))
    (d / "config.json").write_text(json.dumps({"noises": [0.0, 3.0], "n_train_steps": 3}))


def test_q2_answer_only_scores_only_answer_tokens(tmp_path):
    fake_q2(tmp_path, [False] * 30 + [True] * 10)
    m = load(TS / "q2_verifier_quality/analyze_q2.py").analyze(tmp_path)
    a = np.load(tmp_path / "arrays.npz")
    assert m["answer_only"]["n_tokens"] == 10
    assert np.isclose(m["answer_only"]["single_sample_ce"], -a["logp"][0, 30:].mean())


def test_q2_answer_only_with_no_answer_tokens_is_empty(tmp_path):
    fake_q2(tmp_path, [False] * 40)
    assert load(TS / "q2_verifier_quality/analyze_q2.py").analyze(tmp_path)["answer_only"] == {"n_tokens": 0}


def test_q3_without_part_b_writes_part_a_and_answer_only(tmp_path):
    fake_q3(tmp_path, [False] * 30 + [True] * 10)
    m = load(TS / "q3_energy_landscape/analyze_q3.py").analyze(tmp_path)
    assert "part_b" not in m
    ao = m["part_a_noise0"]["answer_only"]
    assert ao["n_tokens"] == 10 and set(ao["ce_after_k_steps"]) == set(range(1, 7))
