import numpy as np
import pytest

from table3_protocol import lightning_epoch_ppl, token_ppl


def test_token_ppl_is_exp_of_the_token_mean():
    assert token_ppl(np.array([2.0, 4.0]), np.array([1, 3])) == pytest.approx(np.exp(6 / 4))


def test_epoch_ppl_is_the_batch_size_weighted_mean_of_batch_ppl():
    ce, n = np.array([1.0, 3.0, 8.0]), np.array([1, 1, 2])
    # batches of 2 in dataset order: {0, 1} -> exp(4/2), {2} -> exp(8/2); weights 2 and 1 sequences
    assert lightning_epoch_ppl(ce, n, batch_size=2) == pytest.approx((2 * np.exp(2) + np.exp(4)) / 3)


def test_epoch_ppl_with_batch_1_is_the_mean_sequence_ppl():
    ce, n = np.array([1.0, 6.0]), np.array([1, 2])
    assert lightning_epoch_ppl(ce, n, batch_size=1) == pytest.approx((np.exp(1) + np.exp(3)) / 2)


def test_epoch_ppl_ddp_interleaves_and_pads_like_distributed_sampler():
    ce, n = np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])
    # 2 ranks: padded to [0, 1, 2, 0]; rank 0 gets [0, 2], rank 1 gets [1, 0]; one batch of 2 each
    expected = (2 * np.exp(2.0) + 2 * np.exp(1.5)) / 4
    assert lightning_epoch_ppl(ce, n, batch_size=2, world_size=2) == pytest.approx(expected)


def test_epoch_ppl_is_never_below_token_ppl():  # Jensen
    rng = np.random.default_rng(0)
    n = rng.integers(5, 200, 100)
    ce = n * rng.uniform(2, 6, 100)
    assert lightning_epoch_ppl(ce, n, batch_size=8) >= token_ppl(ce, n)
