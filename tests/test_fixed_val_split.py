import numpy as np
import pytest

from data.fixed_val_split import fixed_val_split


def _save(tmp_path, idx):
    path = tmp_path / "val.npy"
    np.save(path, np.asarray(idx, dtype=np.int64))
    return str(path)


def test_split_is_disjoint_complete_and_keeps_val_order(tmp_path):
    full = list(range(10))
    train, val = fixed_val_split(full, _save(tmp_path, [7, 2, 5]))
    assert [val[i] for i in range(len(val))] == [7, 2, 5]
    assert sorted(train[i] for i in range(len(train))) == [0, 1, 3, 4, 6, 8, 9]


def test_indices_are_numpy_not_python_lists(tmp_path):
    train, val = fixed_val_split(list(range(10)), _save(tmp_path, [1, 3]))
    assert isinstance(train.indices, np.ndarray) and isinstance(val.indices, np.ndarray)


@pytest.mark.parametrize("bad", [[1, 1], [10], [-1], [[1, 2]]])
def test_rejects_bad_indices(tmp_path, bad):
    with pytest.raises(ValueError):
        fixed_val_split(list(range(10)), _save(tmp_path, bad))
