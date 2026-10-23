"""Validation every N global steps across epoch boundaries, and a resume that lands in epoch 1.

640 train docs at batch 32 on one GPU give 20 steps per epoch, so --max_steps 50 crosses two epoch
boundaries and --val_check_interval 15 must validate after global steps 15, 30 and 45.
"""
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SMOKE_CACHE = REPO / "experiments/scale_to_paper_level/data/smoke/pajama_subset/raw_cache"
STEPS = 50

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(not SMOKE_CACHE.is_dir(), reason="needs the smoke build's download cache"),
]


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    """700-doc pajama_subset reusing the smoke download cache, plus 60 validation indices (640 train docs)."""
    root = tmp_path_factory.mktemp("cadence")
    (root / "data/pajama_subset").mkdir(parents=True)
    (root / "data/pajama_subset/raw_cache").symlink_to(SMOKE_CACHE)
    subprocess.run([sys.executable, "-m", "data.nlp.pajama_subset", "--dataset_dir", str(root / "data"),
                    "--num_docs", "700", "--download_workers", "1"], cwd=REPO, check=True)
    val = root / "val60.npy"
    np.save(val, np.sort(np.random.default_rng(0).choice(700, 60, replace=False)).astype(np.int64))
    return root, val


def _train(root, val, run_name, *extra):
    args = [sys.executable, str(REPO / "train_model.py"), "--model_name", "baseline_transformer",
            "--modality", "NLP", "--model_size", "xxs", "--pretokenize_dataset", "--tokenizer", "EleutherAI/gpt-neox-20b",
            "--context_length", "256", "--ffn_dim_multiplier", "1", "--gpus", "1", "--distributed_strategy", "ddp",  # DistributedSampler, as in the real runs
            "--peak_learning_rate", "0.0012", "--batch_size_per_device", "32", "--accumulate_grad_batches", "1",
            "--gradient_clip_val", "1.0", "--weight_decay", "0.01", "--min_lr_scale", "10", "--warm_up_steps", "10",
            "--max_scheduling_steps", str(STEPS), "--dataset_name", "pajama_subset", "--dataset_dir", str(root / "data"),
            "--fixed_val_indices", str(val), "--val_check_interval", "15", "--check_val_every_n_epoch", "0",
            "--num_workers", "2", "--save_top_k_ckpts", "-1", "--log_every_n_steps", "1", "--no_wandb",
            "--set_matmul_precision", "medium", "--run_name", run_name, *extra]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}  # GPUs 4-5 belong to other users
    subprocess.run(args, cwd=root, env=env, check=True, stdout=subprocess.DEVNULL)


def _metrics(root, run_name):
    """(validation steps, {step: train loss}) from every CSV version of the run."""
    val_steps, train_loss = [], {}
    for path in sorted((root / "logs/csv" / run_name).glob("version_*/metrics.csv")):
        for row in csv.DictReader(open(path)):
            if row.get("valid_loss"):
                val_steps.append(int(row["step"]))
            if row.get("train_loss"):
                train_loss[int(row["step"])] = float(row["train_loss"])
    return val_steps, train_loss


def _ckpt_steps(root, run_name):
    names = [p.name for p in (root / "logs/checkpoints").glob(f"{run_name}_*/*.ckpt")]
    return {int(m.group(1)) for n in names if (m := re.search(r"step=(\d+)", n))}


def test_validation_every_15_global_steps_across_epochs(tiny_data):
    root, val = tiny_data
    _train(root, val, "full", "--max_steps", str(STEPS))
    val_steps, _ = _metrics(root, "full")
    assert val_steps == [14, 29, 44]  # 0-based global steps of the validations, across 2 epoch boundaries
    assert _ckpt_steps(root, "full") == {15, 30, 45}


def test_resume_in_epoch_1_continues_the_stream(tiny_data):
    # checkpoints exist only at validations, so a real resume starts from one: step 30 is 10 steps into epoch 1
    root, val = tiny_data
    ckpt = next((root / "logs/checkpoints").glob("full_*/*step=30-*.ckpt"))
    _train(root, val, "resumed", "--max_steps", str(STEPS), "--resume_training_ckpt", str(ckpt),
           "--resume_data_offset_steps", "30")
    val_steps, resumed = _metrics(root, "resumed")
    # the checkpoint is written inside the step-30 validation, so Lightning repeats that validation on restore
    assert val_steps == [29, 44]
    _, uninterrupted = _metrics(root, "full")
    # same weights at step 30 and the same batches for the rest of epoch 1 -> the same losses up to GPU
    # nondeterminism (a wrong batch differs by far more than 0.02 nats); epoch 2 is reordered by design
    after = [s for s in sorted(resumed) if 30 <= s < 40]
    assert after == list(range(30, 40)), after
    for s in after:
        assert resumed[s] == pytest.approx(uninterrupted[s], abs=0.02), s
