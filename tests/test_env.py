"""Environment, channel, and observation tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from envs.channel import packet_success_probability
from envs.formation_env import EnvConfig, ObsIndex, UAVFormationEnv
from envs.geometry import desired_slot


def _quiet_cfg(**kwargs) -> EnvConfig:
    """Open-field config so unit tests are about one mechanism at a time."""
    base = dict(
        n_followers=1,
        max_steps=20,
        start_noise=0.0,
        start_xy=(150.0, 400.0),
        start_speed=2.0,
        randomize_channel=False,
        randomize_wind=False,
        wind_speed=0.0,
        obstacles=[],
        extra_obstacles=[],
        max_obstacles=1,
        n_nearest_obstacles=1,
        waypoints=[(900.0, 400.0)],
        max_waypoints=1,
        channel_mode="independent",
        p0=0.0,
        obs_mode="hold",
        use_aoi=False,
        use_blockage=False,
        sensor_range=10_000.0,
    )
    base.update(kwargs)
    return EnvConfig(**base)


def test_packet_success_formulas():
    assert packet_success_probability(999.0, 0.2, 10.0, "perfect") == 1.0
    assert packet_success_probability(0.0, 0.4, 100.0, "independent") == pytest.approx(0.4)
    assert packet_success_probability(500.0, 0.4, 100.0, "independent") == pytest.approx(0.4)
    near = packet_success_probability(0.0, 1.0, 100.0, "distance")
    far = packet_success_probability(300.0, 1.0, 100.0, "distance")
    assert near == pytest.approx(1.0)
    assert far == pytest.approx(math.exp(-3.0))
    assert far < near
    with pytest.raises(ValueError):
        packet_success_probability(1.0, 0.5, 10.0, "wifi")


def test_reset_and_step_shapes():
    env = UAVFormationEnv(EnvConfig(n_followers=4, randomize_channel=False))
    obs, info = env.reset(seed=0)
    assert len(obs) == 5
    assert obs[0].shape == (ObsIndex.dim(env.cfg.n_nearest_obstacles),)
    assert env.global_state().shape == (env.global_dim,)
    assert obs[0][ObsIndex.IS_LEADER] == pytest.approx(1.0)
    assert obs[1][ObsIndex.IS_LEADER] == pytest.approx(0.0)
    assert info["aoi"].shape == (4,)
    actions = np.zeros((5, 2), dtype=np.float32)
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    assert len(next_obs) == 5
    assert rewards.shape == (5,)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    # The three ending flags partition an episode. This step may not end it.
    if info["episode"] is not None:
        ep = info["episode"]
        assert ep["success"] + ep["collision"] + ep["timeout"] == pytest.approx(1.0)


def test_obstacle_feature_present_and_absent():
    env = UAVFormationEnv(_quiet_cfg(obstacles=[{"x": 420.0, "y": 300.0, "radius": 40.0}]))
    obs, _ = env.reset(seed=0)
    present = ObsIndex.OBS_START + ObsIndex.OBS_FEATS - 1
    assert obs[0][present] == pytest.approx(1.0)
    # Relative obstacle position matches the leader pose.
    leader = env.uavs[0]
    rel_x = (420.0 - leader.x) / env.cfg.map_width
    assert obs[0][ObsIndex.OBS_START] == pytest.approx(rel_x)

    empty = UAVFormationEnv(_quiet_cfg())
    obs, _ = empty.reset(seed=0)
    assert obs[0][ObsIndex.OBS_START + ObsIndex.OBS_FEATS - 1] == pytest.approx(0.0)


def test_aoi_increments_caps_and_can_be_hidden():
    env = UAVFormationEnv(
        _quiet_cfg(obs_mode="aoi", use_aoi=True, aoi_cap=4, p0=0.0, max_steps=10)
    )
    obs, info = env.reset(seed=0)
    assert info["aoi"][0] == 0
    assert obs[1][ObsIndex.AOI] == pytest.approx(0.0)
    for expected in range(1, 6):
        obs, _, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))
        assert not terminated and not truncated
        assert env.links[0].aoi == min(expected, 4)
        assert obs[1][ObsIndex.AOI] == pytest.approx(min(expected, 4) / 4.0)

    hidden = UAVFormationEnv(_quiet_cfg(obs_mode="aoi", use_aoi=False, p0=0.0))
    hidden.reset(seed=0)
    obs, _, _, _, _ = hidden.step(np.zeros((2, 2), dtype=np.float32))
    assert hidden.links[0].aoi == 1
    assert obs[1][ObsIndex.AOI] == pytest.approx(0.0)


def test_perfect_obs_ignores_packet_loss():
    env = UAVFormationEnv(_quiet_cfg(obs_mode="perfect", channel_mode="independent", p0=0.0))
    env.reset(seed=0)
    env.uavs[0].v = 2.0
    env.uavs[0].heading = 0.0
    env.uavs[1].v = 0.0
    env.step(np.zeros((2, 2), dtype=np.float32))
    ref = env.reference_leader_state(0)
    assert env.links[0].aoi == 1  # the link really did fail
    assert ref[0] == pytest.approx(env.uavs[0].x)
    assert ref[1] == pytest.approx(env.uavs[0].y)


def test_hold_keeps_stale_position_and_extrapolate_uses_old_heading():
    env = UAVFormationEnv(_quiet_cfg(obs_mode="hold", p0=0.0))
    env.reset(seed=0)
    x0 = env.uavs[0].x
    y0 = env.uavs[0].y
    env.uavs[0].v = 2.0
    env.uavs[0].heading = 0.0
    env.uavs[1].v = 0.0
    env.links[0].last_x = x0
    env.links[0].last_y = y0
    env.links[0].last_v = 2.0
    env.links[0].last_heading = 0.0
    env.links[0].aoi = 0
    actions = np.zeros((2, 2), dtype=np.float32)
    actions[0, 1] = 1.0  # max heading change
    env.step(actions)
    ref = env.reference_leader_state(0)
    assert env.links[0].aoi == 1
    assert ref[0] == pytest.approx(x0)
    assert ref[1] == pytest.approx(y0)
    assert env.uavs[0].x != pytest.approx(x0)

    extra = UAVFormationEnv(_quiet_cfg(obs_mode="extrapolate", p0=0.0))
    extra.reset(seed=0)
    x0 = extra.uavs[0].x
    y0 = extra.uavs[0].y
    extra.uavs[0].v = 2.0
    extra.uavs[0].heading = 0.0
    extra.uavs[1].v = 0.0
    extra.links[0].last_x = x0
    extra.links[0].last_y = y0
    extra.links[0].last_v = 2.0
    extra.links[0].last_heading = 0.0
    extra.links[0].aoi = 0
    actions = np.zeros((2, 2), dtype=np.float32)
    actions[0, 1] = 1.0
    extra.step(actions)
    ref = extra.reference_leader_state(0)
    # Age is 1 and the stored heading is still 0, so the prediction walks
    # straight even though the true leader has started to turn.
    assert ref[0] == pytest.approx(x0 + 2.0 * extra.cfg.dt * 1)
    assert ref[1] == pytest.approx(y0)
    assert ref[0] != pytest.approx(extra.uavs[0].x)


def test_perfect_channel_keeps_aoi_at_zero():
    env = UAVFormationEnv(
        _quiet_cfg(channel_mode="perfect", p0=0.0, obs_mode="aoi", use_aoi=True, max_steps=6)
    )
    env.reset(seed=0)
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(np.zeros((2, 2), dtype=np.float32))
        assert not terminated and not truncated
        assert info["aoi"][0] == 0


def test_obstacle_and_inter_uav_collisions():
    env = UAVFormationEnv(
        _quiet_cfg(
            obstacles=[{"x": 300.0, "y": 300.0, "radius": 30.0}],
            max_steps=5,
        )
    )
    env.reset(seed=0)
    for uav in env.uavs:
        uav.v = 0.0
    env.uavs[0].x = 300.0
    env.uavs[0].y = 300.0
    env.uavs[1].x = 100.0
    env.uavs[1].y = 100.0
    _, _, terminated, _, info = env.step(np.zeros((2, 2), dtype=np.float32))
    assert terminated
    assert info["episode"]["obstacle_collision"] == 1.0
    assert info["episode"]["inter_uav_collision"] == 0.0
    assert info["episode"]["success"] == 0.0

    env = UAVFormationEnv(_quiet_cfg(max_steps=5))
    env.reset(seed=0)
    for uav in env.uavs:
        uav.v = 0.0
        uav.x = 300.0
        uav.y = 300.0
    _, _, terminated, _, info = env.step(np.zeros((2, 2), dtype=np.float32))
    assert terminated
    assert info["episode"]["inter_uav_collision"] == 1.0


def test_moving_obstacle_bounces_inside_the_map():
    env = UAVFormationEnv(
        _quiet_cfg(
            obstacles=[{"x": 100.0, "y": 100.0, "radius": 10.0, "vx": 0.0, "vy": 3.0}],
            max_steps=5,
        )
    )
    env.reset(seed=0)
    y0 = env.obstacles[0].y
    env.step(np.zeros((2, 2), dtype=np.float32))
    assert env.obstacles[0].y == pytest.approx(y0 + 3.0 * env.cfg.dt)


def test_channel_randomization_changes_p0_and_option_overrides_it():
    env = UAVFormationEnv(
        EnvConfig(
            n_followers=1,
            randomize_channel=True,
            p0_low=0.2,
            p0_high=0.9,
            channel_mode="distance",
            start_noise=0.0,
            obstacles=[],
            max_obstacles=1,
        )
    )
    draws = []
    for _ in range(12):
        env.reset()
        draws.append(env.ep_p0)
    assert min(draws) >= 0.2 - 1e-9
    assert max(draws) <= 0.9 + 1e-9
    assert len({round(p, 5) for p in draws}) > 1
    env.reset(options={"p0": 0.33, "randomize_channel": True})
    assert env.ep_p0 == pytest.approx(0.33)


def test_same_seed_replays_the_same_episode():
    def once(seed: int):
        env = UAVFormationEnv(EnvConfig(n_followers=2, randomize_channel=True))
        obs, _ = env.reset(seed=seed)
        actions = np.array([[0.2, -0.1], [0.0, 0.3], [-0.4, 0.2]], dtype=np.float32)
        obs2, rewards, _, _, info = env.step(actions)
        return obs[1].copy(), obs2[1].copy(), rewards.copy(), info["aoi"].copy()

    a = once(7)
    b = once(7)
    for left, right in zip(a, b):
        assert np.allclose(left, right)


def test_desired_slot_rotates_with_heading():
    slot = desired_slot((10.0, 20.0), math.pi / 2, (-4.0, 0.0))
    # Facing +y, "behind" is -world y.
    assert slot[0] == pytest.approx(10.0)
    assert slot[1] == pytest.approx(16.0)


def test_building_on_the_segment_is_nlos_and_cuts_delivery():
    open_p = packet_success_probability(100.0, 1.0, 200.0, "distance", blocked=False, nlos_factor=0.25)
    blocked_p = packet_success_probability(100.0, 1.0, 200.0, "distance", blocked=True, nlos_factor=0.25)
    assert blocked_p == pytest.approx(open_p * 0.25)
    # The distance-independent ablation ignores buildings on purpose.
    assert packet_success_probability(100.0, 0.4, 200.0, "independent", blocked=True) == pytest.approx(0.4)

    env = UAVFormationEnv(
        _quiet_cfg(
            use_blockage=True,
            channel_mode="distance",
            p0=1.0,
            d0=1.0e6,
            nlos_factor=0.25,
            obstacles=[{"x": 200.0, "y": 400.0, "radius": 20.0}],
            start_xy=(100.0, 400.0),
            start_noise=0.0,
        )
    )
    env.reset(seed=0)
    env.uavs[0].x, env.uavs[0].y = 100.0, 400.0
    env.uavs[1].x, env.uavs[1].y = 300.0, 400.0
    assert env.link_blocked(0)
    env.uavs[1].y = 480.0
    assert not env.link_blocked(0)


def test_far_obstacles_fall_outside_the_sensor():
    env = UAVFormationEnv(
        _quiet_cfg(
            sensor_range=40.0,
            obstacles=[{"x": 900.0, "y": 400.0, "radius": 20.0}],
            start_xy=(150.0, 400.0),
            start_noise=0.0,
        )
    )
    obs, _ = env.reset(seed=0)
    present = ObsIndex.OBS_START + ObsIndex.OBS_FEATS - 1
    assert obs[0][present] == pytest.approx(0.0)


def test_default_route_is_a_cluttered_transit():
    from envs.geometry import segment_hits_disk

    env = UAVFormationEnv(
        EnvConfig(randomize_channel=False, randomize_wind=False, start_noise=0.0)
    )
    env.reset(seed=0)
    assert len(env.obstacles) >= 8
    movers = [o for o in env.obstacles if o.vx != 0.0 or o.vy != 0.0]
    assert len(movers) >= 2
    blocked_legs = 0
    for (ax, ay), (bx, by) in zip(env.waypoints[:-1], env.waypoints[1:]):
        if any(
            segment_hits_disk(ax, ay, bx, by, o.x, o.y, o.radius) for o in env.obstacles
        ):
            blocked_legs += 1
    assert blocked_legs >= 2
    for wx, wy in env.waypoints:
        for o in env.obstacles:
            assert math.hypot(wx - o.x, wy - o.y) > o.radius + env.cfg.wp_radius
    _, _, _, _, info = env.step(np.zeros((env.n_agents, 2), dtype=np.float32))
    assert info["collision"] is False


def test_bad_action_shape_and_unknown_config_key():
    env = UAVFormationEnv(_quiet_cfg())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros((1, 2)))
    with pytest.raises(KeyError):
        from envs.formation_env import env_config_from_dict

        env_config_from_dict({"not_a_real_key": 1})
