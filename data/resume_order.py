"""Mid-epoch resume for a non-resumable DataLoader: reorder the train set so a fresh sampler continues where the interrupted run stopped."""
import math

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def resume_position(global_steps: int, n_train: int, per_device_batch: int, world_size: int) -> tuple[int, int]:
    """Map optimizer steps already trained to (epoch, samples of that epoch already seen).

    DistributedSampler pads each rank to ceil(n / world_size) samples and the loader keeps the last partial
    batch (drop_last=False), so an epoch has ceil(ceil(n / world_size) / per_device_batch) steps.
    """
    steps_per_epoch = math.ceil(math.ceil(n_train / world_size) / per_device_batch)
    epoch, step_in_epoch = divmod(global_steps, steps_per_epoch)
    return epoch, step_in_epoch * per_device_batch * world_size


def _perm(n: int, seed: int) -> np.ndarray:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n, generator=g).numpy()


def continue_epoch_order(ds: Dataset, offset: int, seed: int, world_size: int = 1, epoch: int = 0,
                         sampler_epoch: int | None = None) -> Subset:
    """Return a same-length view of ds whose shuffled order starts at sample `offset` of the original `epoch` order.

    DistributedSampler(shuffle=True, seed=seed) visits perm = randperm(n) seeded with seed + epoch and gives rank r the
    positions perm[r::world_size]. With q the permutation of the sampler that reads the view (seeded with
    seed + sampler_epoch, default `epoch`), mapping position q[j] to ds[perm[(j + offset) % n]] makes that sampler
    yield the original stream minus its first `offset` samples. The length is unchanged, so Lightning's
    batches-per-epoch count still matches the checkpoint's progress.
    """
    n = len(ds)
    if not 0 <= offset < n or offset % world_size:
        raise ValueError(f"offset must be in [0, {n}) and a multiple of world_size={world_size}, got {offset}")
    perm = _perm(n, seed + epoch)
    reader = perm if sampler_epoch is None or sampler_epoch == epoch else _perm(n, seed + sampler_epoch)
    order = np.empty(n, dtype=np.int64)
    order[reader] = np.roll(perm, -offset)
    return Subset(ds, order)
