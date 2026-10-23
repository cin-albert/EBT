import numpy as np

from make_val_indices import make_val_indices


def test_val_indices_unique_sorted_in_range():
    idx = make_val_indices(1_000_000)
    assert len(idx) == 33_000 and idx.dtype == np.int64
    assert np.all(np.diff(idx) > 0) and idx[0] >= 0 and idx[-1] < 1_000_000


def test_val_indices_deterministic():
    assert np.array_equal(make_val_indices(10**6), make_val_indices(10**6))


def test_val_indices_cover_whole_range():  # not clustered in the first shards
    idx = make_val_indices(1_000_000)
    assert (idx >= 500_000).mean() > 0.45
