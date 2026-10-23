"""Train/validation split with validation indices fixed by a file (keeps a validation set across dataset rebuilds)."""
import numpy as np
from torch.utils.data import Dataset, Subset


def fixed_val_split(full_ds: Dataset, val_indices_path: str) -> tuple[Subset, Subset]:
    """Return (train, val); val keeps the file order, train is every other index in ascending order.

    Indices stay numpy arrays so forked DataLoader workers share them instead of copying Python ints.
    """
    val_idx = np.load(val_indices_path)
    n = len(full_ds)
    if val_idx.ndim != 1 or len(np.unique(val_idx)) != len(val_idx) or val_idx.min() < 0 or val_idx.max() >= n:
        raise ValueError(f"{val_indices_path}: need unique 1-D indices in [0, {n}), got shape {val_idx.shape}")
    is_val = np.zeros(n, dtype=bool)
    is_val[val_idx] = True
    return Subset(full_ds, np.flatnonzero(~is_val)), Subset(full_ds, val_idx.astype(np.int64))
