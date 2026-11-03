"""Unit tests for capture-latch vs frozen simultaneous capture semantics."""
import numpy as np
import pytest

from uav_planning.v4_capture_latch import (
    CAPTURE_SPEED_MPS,
    GOAL_BALL_M,
    V4MissionCaptureVariant,
    capture_mask,
    make_standard_env,
)
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import default_config


def _empty_env(latch):
    cfg = default_config('urban_canyon', 'L0', density=0., sensor_range=400.)
    env = V4MissionCaptureVariant(cfg, capture_latch=latch)
    env.reset(101)
    return env


def test_capture_mask_thresholds():
    goals = np.zeros((3, 3))
    pos = np.array([[3.9, 0, 0], [4.0, 0, 0], [0, 0, 0]], dtype=float)
    vel = np.array([[1.9, 0, 0], [0, 0, 0], [2.0, 0, 0]], dtype=float)
    mask = capture_mask(pos, goals, vel)
    assert mask.tolist() == [True, False, False]
    assert GOAL_BALL_M == 4.0 and CAPTURE_SPEED_MPS == 2.0


def test_baseline_matches_frozen_v4mission_on_same_actions():
    cfg = default_config('urban_canyon', 'L0', density=0., sensor_range=400.)
    frozen = V4Mission(cfg)
    variant = V4MissionCaptureVariant(cfg, capture_latch=False)
    o1 = frozen.reset(202)
    o2 = variant.reset(202)
    np.testing.assert_allclose(o1, o2)
    rng = np.random.default_rng(0)
    for _ in range(40):
        action = rng.uniform(-1, 1, size=(3, 3))
        o1, r1, d1, t1, i1 = frozen.step(action)
        o2, r2, d2, t2, i2 = variant.step(action)
        np.testing.assert_allclose(o1, o2, atol=1e-6)
        assert r1 == pytest.approx(r2, rel=0, abs=1e-6)
        assert d1 == d2 and t1 == t2
        assert i1['success'] == i2['success']
        assert i1['arrived_by_uav'] == i2['arrived_by_uav']
        np.testing.assert_array_equal(frozen.captured, variant.captured)
        if d1 or t1:
            break


def test_latch_stays_captured_after_leave_and_skips_leave_penalty():
    env = _empty_env(latch=True)
    # Teleport all agents into ball at low speed to rise-edge capture.
    env.p = env.goal.copy()
    env.v = np.zeros_like(env.v)
    before = env.captured.copy()
    assert not before.any()
    # One no-op-ish action near zero accel
    obs, reward, done, truncated, info = env.step(np.zeros((3, 3)))
    assert env.captured.all()
    assert info['capture_latch'] is True
    assert info['success'] is True or info['arrived_by_uav'] == [True, True, True]
    # Force leave: move far and fast; latch must keep captured True and not -5-spam.
    # After success the episode may already be done; reset a fresh latch env and
    # manually set state to test leave without terminating via success bonus path.
    env = _empty_env(latch=True)
    env.p = env.goal.copy()
    env.v[:] = 0
    env.captured[:] = True  # already latched
    # Move outside ball at high speed in one step with strong action
    # First zero-action keeps them captured without needing rising edge.
    env.p = env.goal + np.array([[20.0, 0, 0]] * 3)
    env.v[:] = 5.0
    prev = env.captured.copy()
    # Craft step internals by calling step with action that won't collide peers
    # Restore positions just outside goals with outward velocity via direct step algebra:
    # Use a controlled call: set positions near goals then apply action.
    env.p = env.goal + np.array([[10.0, 0.0, 0.0]] * 3)
    env.v = np.array([[5.0, 0.0, 0.0]] * 3)
    env.captured[:] = True
    # Zero action → limited() keeps high speed; positions move further out.
    obs, reward, done, truncated, info = env.step(np.zeros((3, 3)))
    assert env.captured.all(), 'latch must remain True after leaving ball'
    assert info['arrived_now_by_uav'] == [False, False, False]
    assert info['arrived_by_uav'] == [True, True, True]
    # Reward should not include three -5 leave penalties (mean would drop hard).
    # With latch, delta is zeros so no -5 from capture channel.
    # Compare to baseline leave penalty magnitude on same geometry.
    base = _empty_env(latch=False)
    base.p = env.goal + np.array([[10.0, 0.0, 0.0]] * 3)
    base.v = np.array([[5.0, 0.0, 0.0]] * 3)
    base.captured[:] = True
    _, reward_base, _, _, info_base = base.step(np.zeros((3, 3)))
    assert not base.captured.any()
    assert info_base['arrived_by_uav'] == [False, False, False]
    # Baseline mean reward is lower by ~5 from the capture deltas (-1 each UAV).
    assert reward > reward_base + 3.0


def test_baseline_uncaptures_on_leave():
    env = _empty_env(latch=False)
    env.p = env.goal.copy()
    env.v[:] = 0
    env.step(np.zeros((3, 3)))
    assert env.captured.all()
    env.p = env.goal + np.array([[12.0, 0, 0]] * 3)
    env.v = np.array([[6.0, 0, 0]] * 3)
    env.step(np.zeros((3, 3)))
    assert not env.captured.any()


def test_make_standard_env_flag():
    env = make_standard_env('warehouse', capture_latch=True)
    assert env.capture_latch is True
    env.reset(830002)
    assert env.config.family == 'warehouse'
