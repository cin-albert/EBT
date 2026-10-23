import pytest
from torch.utils.data import DistributedSampler

from data.resume_order import continue_epoch_order, resume_position

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


def test_resume_position_epoch_boundary():
    # 1000 docs, 2 ranks x 8 -> ceil(500/8) = 63 steps per epoch
    assert resume_position(63, 1000, 8, 2) == (1, 0)
    assert resume_position(70, 1000, 8, 2) == (1, 7 * 16)


def test_resume_position_epoch_0_unchanged():
    assert resume_position(25, 1000, 8, 2) == (0, 25 * 16)


def test_continue_epoch_order_matches_sampler_in_epoch_1():
    full = list(range(N))
    s = DistributedSampler(full, num_replicas=1, rank=0, shuffle=True, seed=SEED)
    s.set_epoch(1)
    expected = list(s)[40:]
    view = continue_epoch_order(full, 40, SEED, epoch=1)
    s2 = DistributedSampler(view, num_replicas=1, rank=0, shuffle=True, seed=SEED)
    s2.set_epoch(1)
    assert [view[i] for i in s2][: N - 40] == expected


def test_view_for_epoch_1_read_by_an_epoch_0_sampler():
    # Lightning 2.4 builds the first epoch after a mid-epoch restart with the sampler still at epoch 0
    full = list(range(N))
    s = DistributedSampler(full, num_replicas=1, rank=0, shuffle=True, seed=SEED)
    s.set_epoch(1)
    expected = list(s)[40:]
    view = continue_epoch_order(full, 40, SEED, epoch=1, sampler_epoch=0)
    s2 = DistributedSampler(view, num_replicas=1, rank=0, shuffle=True, seed=SEED)
    assert [view[i] for i in s2][: N - 40] == expected
