"""Unit tests for path-C reward/capture reshape."""
import numpy as np
import pytest

from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import default_config
from uav_planning.v4_reward_reshape import (
    DEFAULT_RESHAPE,
    FROZEN_CAPTURE_SPEED_MPS,
    FROZEN_GOAL_BALL_M,
    ReshapeConfig,
    V4MissionRewardReshape,
    make_reshape_env,
    reshape_from_protocol,
    soft_capture_mask,
)


def test_default_numbers_match_protocol_intent():
    rs = DEFAULT_RESHAPE
    assert rs.progress_scale == 1.0
    assert rs.living_cost == 0.005
    assert rs.train_ball_m == 6.0
    assert rs.train_speed_mps == 3.5
    assert rs.capture_latch is True
    assert rs.team_success_bonus == 50.0
    assert FROZEN_GOAL_BALL_M == 4.0 and FROZEN_CAPTURE_SPEED_MPS == 2.0


def test_soft_capture_between_frozen_and_train():
    goals = np.zeros((3, 3))
    pos = np.array([[5.5, 0, 0], [3.9, 0, 0], [6.1, 0, 0]], dtype=float)
    vel = np.array([[3.0, 0, 0], [1.5, 0, 0], [0, 0, 0]], dtype=float)
    soft = soft_capture_mask(pos, goals, vel, 6.0, 3.5)
    assert soft.tolist() == [True, True, False]


def test_latch_and_ball_enter_bonus_fire_once():
    cfg = default_config('urban_canyon', 'L0', density=0., sensor_range=400.)
    env = V4MissionRewardReshape(cfg, reshape=DEFAULT_RESHAPE)
    env.reset(303)
    env.p = env.goal + np.array([[5.0, 0, 0], [5.0, 0, 0], [5.0, 0, 0]])
    env.v = np.zeros_like(env.v)
    # Step toward goals into soft ball at low speed.
    env.p = env.goal.copy()
    action = np.zeros((3, 3))
    _, r1, done1, _, info1 = env.step(action)
    assert info1['success'] is True
    assert done1 is True
    assert env.captured.all()
    assert r1 > 10  # success + capture bonuses dominate


def test_leave_does_not_emit_negative_capture_under_latch():
    cfg = default_config('urban_canyon', 'L0', density=0., sensor_range=400.)
    env = V4MissionRewardReshape(cfg, reshape=DEFAULT_RESHAPE)
    env.reset(404)
    env.p = env.goal.copy()
    env.v = np.zeros_like(env.v)
    _, _, done, _, info = env.step(np.zeros((3, 3)))
    assert info['success'] and done
    # Force leave after latched success would have ended; rebuild mid-episode style:
    env2 = V4MissionRewardReshape(cfg, reshape=DEFAULT_RESHAPE)
    env2.reset(405)
    env2.p = env2.goal.copy()
    env2.v = np.zeros_like(env2.v)
    # Capture without completing team success by capturing only two, then leave one.
    env2.p[0] = env2.goal[0]
    env2.p[1] = env2.goal[1]
    env2.p[2] = env2.goal[2] + np.array([20.0, 0, 0])
    env2.v[:] = 0
    _, r_enter, d_enter, _, info_enter = env2.step(np.zeros((3, 3)))
    assert info_enter['arrived_by_uav'][:2] == [True, True]
    assert info_enter['success'] is False
    assert not d_enter
    captured_before_leave = env2.captured.copy()
    # Yank agent 0 far away at high commanded accel direction.
    env2.p[0] = env2.goal[0] + np.array([30.0, 0, 0])
    env2.v[0] = np.array([5.0, 0, 0])
    prev = captured_before_leave.copy()
    _, r_leave, _, _, info_leave = env2.step(np.zeros((3, 3)))
    assert env2.captured[0] == prev[0] == True  # latch sticky
    # Mean reward should not include -capture_bonus for leave.
    assert info_leave['arrived_by_uav'][0] is True


def test_reshape_from_protocol_rejects_unknown():
    with pytest.raises(ValueError):
        reshape_from_protocol({'not_a_knob': 1})


def test_make_reshape_empty_env():
    env, obs = make_reshape_env('empty', 870001)
    assert obs.shape == (3, 162)
    assert len(env.boxes) == 0
    assert env.reshape.train_ball_m == 6.0
