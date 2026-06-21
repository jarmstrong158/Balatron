"""Tests for the self-imitation DemoBuffer (Phase 1: capture + persistence)."""
import numpy as np
import pytest

from demo_buffer import DemoBuffer

SD, AD, MD = 838, 14, 45  # state / action / mask dims (mirror live config)


def _traj(n, fill):
    """A trajectory of n transitions whose arrays are all valued `fill`."""
    states = [np.full(SD, fill, dtype=np.float32) for _ in range(n)]
    actions = [np.full(AD, fill, dtype=np.float32) for _ in range(n)]
    masks = [np.full(MD, 1.0, dtype=np.float32) for _ in range(n)]
    heads = [fill % 3 for _ in range(n)]
    return states, actions, masks, heads


def _buf(tmp_path, capacity=100):
    return DemoBuffer(capacity, SD, AD, MD, path=str(tmp_path / "demos.npz"))


def test_empty_buffer(tmp_path):
    b = _buf(tmp_path)
    assert len(b) == 0
    assert b.sample(8) is None


def test_add_trajectory_grows_and_stores(tmp_path):
    b = _buf(tmp_path)
    added = b.add_trajectory(*_traj(5, 1))
    assert added == 5
    assert len(b) == 5
    assert b.trajectories_added == 1
    assert b.transitions_added == 5
    # stored content matches
    assert np.all(b.states[0] == 1.0)
    assert b.head_indices[0] == 1 % 3


def test_empty_trajectory_is_noop(tmp_path):
    b = _buf(tmp_path)
    assert b.add_trajectory([], [], [], []) == 0
    assert len(b) == 0
    assert b.trajectories_added == 0


def test_ring_eviction_at_capacity(tmp_path):
    b = _buf(tmp_path, capacity=10)
    b.add_trajectory(*_traj(8, 1))    # fill 0..7
    b.add_trajectory(*_traj(8, 2))    # wraps: overwrites 0..5, pos=6
    assert len(b) == 10               # capped at capacity
    assert b.full is True
    assert b.transitions_added == 16  # lifetime counter keeps counting
    # the oldest 6 of the first trajectory were overwritten by the second
    assert np.all(b.states[0] == 2.0)
    # slots 6,7 still hold the tail of the first trajectory
    assert np.all(b.states[6] == 1.0)


def test_sample_shapes_and_bounds(tmp_path):
    b = _buf(tmp_path)
    b.add_trajectory(*_traj(20, 7))
    s = b.sample(8)
    assert s["states"].shape == (8, SD)
    assert s["actions"].shape == (8, AD)
    assert s["masks"].shape == (8, MD)
    assert s["head_indices"].shape == (8,)
    # sampling more than available clamps to len
    s2 = b.sample(999)
    assert s2["states"].shape[0] == 20


def test_save_load_roundtrip(tmp_path):
    b = _buf(tmp_path)
    b.add_trajectory(*_traj(12, 3))
    b.add_trajectory(*_traj(4, 9))
    b.save()

    # fresh buffer at same path reloads the persisted demos + lifetime meta
    b2 = _buf(tmp_path)
    assert len(b2) == 16
    assert b2.trajectories_added == 2
    assert b2.transitions_added == 16
    assert np.array_equal(b.states[:16], b2.states[:16])
    assert np.array_equal(b.head_indices[:16], b2.head_indices[:16])


def test_load_truncates_to_smaller_capacity(tmp_path):
    big = _buf(tmp_path, capacity=100)
    big.add_trajectory(*_traj(60, 5))
    big.save()
    # reopening with a smaller capacity must not overflow
    small = DemoBuffer(40, SD, AD, MD, path=str(tmp_path / "demos.npz"))
    assert len(small) == 40
    assert small.full is True
