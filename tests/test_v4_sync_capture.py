import subprocess
import sys

import numpy as np
import pytest

from uav_planning.run_lock import RunLock
from uav_planning.v4_reward_reshape import ReshapeConfig, make_reshape_env
from uav_planning.v4_sync_capture import (
    SyncCaptureConfig, capture_potential, failure_reason, occupancy, shape_reward,
)


def test_default_off_preserves_exact_reward_object():
    reward = np.float32(-0.0)
    result, shaping = shape_reward(reward, None, None, terminated=True,
                                   truncated=True, gamma=0.99, config=SyncCaptureConfig())
    assert result is reward
    assert shaping == 0.0


@pytest.mark.parametrize('terminal', [False, True])
def test_discounted_telescoping_and_timeout_bootstrap(terminal):
    cfg = SyncCaptureConfig(enabled=True)
    gamma = 0.99
    potentials = [0.11, 0.2, 0.73, 0.85]
    changes = [shape_reward(0., potentials[t], potentials[t + 1],
                           terminated=terminal and t == 2, truncated=t == 2,
                           gamma=gamma, config=cfg)[1] for t in range(3)]
    total = sum(gamma ** t * change for t, change in enumerate(changes))
    boundary = 0.0 if terminal else gamma ** 3 * cfg.coefficient * potentials[-1]
    assert total == pytest.approx(-cfg.coefficient * potentials[0] + boundary, abs=1e-14)
    base_bootstrap_value = 7.0
    next_value = 0.0 if terminal else base_bootstrap_value - cfg.coefficient * potentials[-1]
    base_next_value = 0.0 if terminal else base_bootstrap_value
    assert total + gamma ** 3 * next_value == pytest.approx(
        gamma ** 3 * base_next_value - cfg.coefficient * potentials[0], abs=1e-14)


def test_potential_targets_worst_agent_distance_and_speed_and_is_finite():
    cfg = SyncCaptureConfig(enabled=True)
    p = np.zeros((3, 3))
    goal = np.zeros_like(p)
    v = np.zeros_like(p)
    perfect = capture_potential(p, goal, v, cfg)
    p[0, 0] = 8.
    distant = capture_potential(p, goal, v, cfg)
    p[0, 0] = 0.
    v[0, 0] = 4.
    fast = capture_potential(p, goal, v, cfg)
    assert perfect == 1.
    assert 0 < distant < perfect
    assert fast == pytest.approx(distant, abs=1e-8)
    p[0, 0] = 1e6
    assert 0 < capture_potential(p, goal, v, cfg) < distant
    with pytest.raises(ValueError):
        capture_potential(p * np.nan, goal, v, cfg)


def test_frozen_capture_predicate_and_default_off_rollout_parity():
    rs = ReshapeConfig(train_ball_m=4., train_speed_mps=2., capture_latch=False)
    first, obs = make_reshape_env('empty', 1411, reshape=rs)
    second, other_obs = make_reshape_env('empty', 1411, reshape=rs)
    np.testing.assert_array_equal(obs, other_obs)
    rng = np.random.default_rng(411)
    off = SyncCaptureConfig()
    on = SyncCaptureConfig(enabled=True)
    for _ in range(32):
        action = rng.uniform(-1, 1, (3, 3))
        previous = capture_potential(second.p, second.goal, second.v, on)
        o1, r1, d1, t1, i1 = first.step(action)
        o2, r2, d2, t2, i2 = second.step(action)
        reward, _ = shape_reward(r2, None, None, terminated=d2, truncated=t2, gamma=.99, config=off)
        assert reward == r1 and d1 == d2 and t1 == t2 and i1 == i2
        np.testing.assert_array_equal(o1, o2)
        shaped, _ = shape_reward(r2, previous, capture_potential(second.p, second.goal, second.v, on),
                                 terminated=d2, truncated=t2, gamma=.99, config=on)
        assert np.isfinite(shaped)
        if d1 or t1:
            break
    first.p = first.goal.copy()
    first.v[:] = 0.
    assert occupancy(first) == 3
    first.v[0, 0] = 2.
    assert occupancy(first) == 2


def test_failure_labels_distinguish_any_ball_from_all_ball():
    assert failure_reason({'timeout': True, 'ever_in_ball_by_uav': [True, False, False]}) == 'timeout_after_ball_any_only'
    assert failure_reason({'timeout': True, 'ever_in_ball_by_uav': [True] * 3}) == 'timeout_after_ball_all'
    assert failure_reason({'timeout': True, 'ever_in_ball_by_uav': [False] * 3}) == 'timeout_before_ball'


def test_lock_excludes_child_process_and_releases(tmp_path):
    path = tmp_path / 'runner.lock'
    code = ('import sys; from uav_planning.run_lock import RunLock; '
            'lock = RunLock(sys.argv[1], "child"); lock.close()')
    command = [sys.executable, '-c', code, str(path)]
    with RunLock(path, 'parent'):
        print('COMMAND ' + subprocess.list2cmdline(command), flush=True)
        blocked = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert blocked.returncode != 0
        assert 'Another runner holds lock' in blocked.stderr
    print('COMMAND ' + subprocess.list2cmdline(command), flush=True)
    released = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert released.returncode == 0, released.stderr
