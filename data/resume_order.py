"""Mid-epoch resume for a non-resumable DataLoader: reorder the train set so a fresh epoch-0 sampler continues where the interrupted run stopped."""
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def continue_epoch_order(ds: Dataset, offset: int, seed: int, world_size: int = 1) -> Subset:
    """Return a same-length view of ds whose epoch-0 shuffled order starts at sample `offset` of the original order.

    DistributedSampler(shuffle=True, seed=seed) visits perm = randperm(n) at epoch 0 and gives rank r the
    positions perm[r::world_size]. Mapping position perm[j] to ds[perm[(j + offset) % n]] makes a fresh
    sampler yield the original stream minus its first `offset` samples. The length is unchanged, so
    Lightning's batches-per-epoch count still matches the checkpoint's progress.
    """
    n = len(ds)
    if not 0 <= offset < n or offset % world_size:
        raise ValueError(f"offset must be in [0, {n}) and a multiple of world_size={world_size}, got {offset}")
    g = torch.Generator()
    g.manual_seed(seed)  # DistributedSampler seeds with seed + epoch; epoch is 0
    perm = torch.randperm(n, generator=g).numpy()
    order = np.empty(n, dtype=np.int64)
    order[perm] = np.roll(perm, -offset)
    return Subset(ds, order)
