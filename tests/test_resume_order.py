import pytest
from torch.utils.data import DistributedSampler

from data.resume_order import continue_epoch_order

N, WORLD, SEED = 100, 2, 33


def _rank_order(ds, rank):
    return [ds[i] for i in DistributedSampler(ds, num_replicas=WORLD, rank=rank, shuffle=True, seed=SEED)]


@pytest.mark.parametrize("offset", [0, 40, 98])
def test_fresh_sampler_continues_the_original_order(offset):
    full = list(range(N))
    reordered = continue_epoch_order(full, offset, SEED)
    for rank in range(WORLD):
        original_rest = _rank_order(full, rank)[offset // WORLD:]
        assert _rank_order(reordered, rank)[: len(original_rest)] == original_rest


def test_every_sample_appears_once():
    reordered = continue_epoch_order(list(range(N)), 40, SEED)
    assert sorted(reordered[i] for i in range(len(reordered))) == list(range(N))


@pytest.mark.parametrize("offset", [-2, N, 41])
def test_rejects_bad_offset(offset):
    with pytest.raises(ValueError):
        continue_epoch_order(list(range(N)), offset, SEED, world_size=WORLD)
