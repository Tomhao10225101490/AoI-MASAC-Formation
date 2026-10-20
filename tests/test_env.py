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
    actions = np.zeros((5, env.action_dim), dtype=np.float32)
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
        obs, _, terminated, truncated, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
        assert not terminated and not truncated
        assert env.links[0].aoi == min(expected, 4)
        assert obs[1][ObsIndex.AOI] == pytest.approx(min(expected, 4) / 4.0)

    hidden = UAVFormationEnv(_quiet_cfg(obs_mode="aoi", use_aoi=False, p0=0.0))
    hidden.reset(seed=0)
    obs, _, _, _, _ = hidden.step(np.zeros((2, hidden.action_dim), dtype=np.float32))
    assert hidden.links[0].aoi == 1
    assert obs[1][ObsIndex.AOI] == pytest.approx(0.0)


def test_perfect_obs_ignores_packet_loss():
    env = UAVFormationEnv(_quiet_cfg(obs_mode="perfect", channel_mode="independent", p0=0.0))
    env.reset(seed=0)
    env.uavs[0].v = 2.0
    env.uavs[0].heading = 0.0
    env.uavs[1].v = 0.0
    env.step(np.zeros((2, env.action_dim), dtype=np.float32))
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
    env.links[0].meas_x = x0
    env.links[0].meas_y = y0
    env.links[0].meas_z = env.uavs[0].z
    env.links[0].meas_v = 2.0
    env.links[0].meas_heading = 0.0
    env.links[0].meas_vz = 0.0
    env.links[0].aoi = 0
    env.reset_belief_from_stored_measurement(0)
    actions = np.zeros((2, env.action_dim), dtype=np.float32)
    actions[0, 1] = 1.0  # max heading change
    env.step(actions)
    ref = env.reference_leader_state(0)
    assert env.links[0].aoi == 1
    assert ref[0] == pytest.approx(x0)
    assert ref[1] == pytest.approx(y0)
    assert env.uavs[0].x != pytest.approx(x0)

    extra = UAVFormationEnv(_quiet_cfg(obs_mode="extrapolate", p0=0.0, wind_obs_noise=0.0))
    extra.reset(seed=0)
    x0 = extra.uavs[0].x
    y0 = extra.uavs[0].y
    extra.uavs[0].v = 2.0
    extra.uavs[0].heading = 0.0
    extra.uavs[1].v = 0.0
    extra.links[0].meas_x = x0
    extra.links[0].meas_y = y0
    extra.links[0].meas_z = extra.uavs[0].z
    extra.links[0].meas_v = 2.0
    extra.links[0].meas_heading = 0.0
    extra.links[0].meas_vz = 0.0
    extra.links[0].aoi = 0
    extra.reset_belief_from_stored_measurement(0)
    actions = np.zeros((2, extra.action_dim), dtype=np.float32)
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
        _, _, terminated, truncated, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
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
    _, _, terminated, _, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
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
    _, _, terminated, _, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
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
    env.step(np.zeros((2, env.action_dim), dtype=np.float32))
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
        actions = np.array(
            [[0.2, -0.1, 0.0], [0.0, 0.3, 0.1], [-0.4, 0.2, -0.2]], dtype=np.float32
        )
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
    for a, b in zip(env.waypoints[:-1], env.waypoints[1:]):
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]
        if any(
            segment_hits_disk(ax, ay, bx, by, o.x, o.y, o.radius) for o in env.obstacles
        ):
            blocked_legs += 1
    assert blocked_legs >= 2
    buildings = [o for o in env.obstacles if o.height > 0.0]
    assert any(o.height < env.cfg.z_max for o in buildings)
    assert any(o.height > env.cfg.z_max for o in buildings)
    assert all(not o.blocks_radio for o in movers)
    for wp in env.waypoints:
        wx, wy = wp[0], wp[1]
        for o in env.obstacles:
            assert math.hypot(wx - o.x, wy - o.y) > o.radius + env.cfg.wp_radius
    _, _, _, _, info = env.step(np.zeros((env.n_agents, env.action_dim), dtype=np.float32))
    assert info["collision"] is False


def test_roof_clearance_restores_los_and_avoids_collision():
    """A short building blocks cruise altitude and is clear once both aircraft are above it."""
    building = {"x": 200.0, "y": 400.0, "radius": 20.0, "height": 30.0, "blocks_radio": True}
    env = UAVFormationEnv(
        _quiet_cfg(
            use_blockage=True,
            channel_mode="distance",
            p0=1.0,
            d0=1.0e6,
            obstacles=[building],
            start_xy=(100.0, 400.0),
            start_z=20.0,
            z_min=10.0,
            start_noise=0.0,
        )
    )
    env.reset(seed=0)
    env.uavs[0].x, env.uavs[0].y, env.uavs[0].z = 100.0, 400.0, 20.0
    env.uavs[1].x, env.uavs[1].y, env.uavs[1].z = 300.0, 400.0, 20.0
    assert env.link_blocked(0)
    env.uavs[0].z = 80.0
    env.uavs[1].z = 80.0
    assert not env.link_blocked(0)
    # The low aircraft is inside the footprint, so the ray enters the volume
    # even if the other aircraft has already climbed clear.
    env.uavs[0].x, env.uavs[0].z = 190.0, 20.0
    env.uavs[1].x, env.uavs[1].z = 300.0, 80.0
    assert env.link_blocked(0)

    # Parked inside the footprint but above the roof: not a collision.
    over = UAVFormationEnv(
        _quiet_cfg(obstacles=[building], start_z=80.0, z_min=10.0, max_steps=3, start_noise=0.0)
    )
    over.reset(seed=0)
    for uav in over.uavs:
        uav.v = 0.0
        uav.vz = 0.0
    over.uavs[0].x, over.uavs[0].y, over.uavs[0].z = 200.0, 400.0, 80.0
    over.uavs[1].x, over.uavs[1].y, over.uavs[1].z = 100.0, 100.0, 80.0
    _, _, terminated, _, info = over.step(np.zeros((2, over.action_dim), dtype=np.float32))
    assert not terminated
    assert info["collision"] is False


def test_vertical_separation_is_not_an_inter_uav_collision():
    env = UAVFormationEnv(_quiet_cfg(max_steps=3))
    env.reset(seed=0)
    for uav in env.uavs:
        uav.v = 0.0
        uav.vz = 0.0
        uav.x = 300.0
        uav.y = 300.0
    env.uavs[0].z = 40.0
    env.uavs[1].z = 80.0
    _, _, terminated, _, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
    assert not terminated
    assert info["collision"] is False


def test_last_waypoint_waits_for_the_formation():
    env = UAVFormationEnv(
        _quiet_cfg(
            waypoints=[(150.0, 400.0, 40.0)],
            start_xy=(150.0, 400.0),
            start_z=40.0,
            start_noise=0.0,
            start_speed=0.0,
            max_steps=5,
            formation_spacing=32.0,
        )
    )
    env.reset(seed=0)
    for uav in env.uavs:
        uav.v = 0.0
        uav.vz = 0.0
    env.uavs[0].x, env.uavs[0].y, env.uavs[0].z = 150.0, 400.0, 40.0
    env.uavs[1].x, env.uavs[1].y, env.uavs[1].z = 40.0, 400.0, 40.0
    _, _, terminated, _, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
    assert info["waypoints_done"] == 0
    assert not terminated
    # Slot for one follower is one spacing behind a heading of 0.
    env.uavs[1].x = 150.0 - env.cfg.formation_spacing
    env.uavs[1].y = 400.0
    env.uavs[1].z = 40.0
    _, rewards, terminated, _, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
    assert terminated
    assert info["episode"]["success"] == 1.0
    assert rewards[0] > 0.0


def test_waypoint_capture_does_not_bill_the_next_leg():
    env = UAVFormationEnv(
        _quiet_cfg(
            waypoints=[(118.0, 400.0, 40.0), (900.0, 400.0, 40.0)],
            max_waypoints=2,
            start_xy=(100.0, 400.0),
            start_z=40.0,
            start_noise=0.0,
            start_speed=16.0,
            wp_radius=20.0,
            w_progress=1.0,
            w_heading=0.0,
            w_smooth=0.0,
            w_step=0.0,
            w_clearance=0.0,
            w_formation=0.0,
            w_form_progress=0.0,
            formation_bonus=0.0,
            wp_bonus=0.0,
            max_steps=5,
        )
    )
    env.reset(seed=0)
    env.uavs[0].v = 16.0
    env.uavs[0].heading = 0.0
    env.uavs[0].vz = 0.0
    _, rewards, _, _, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
    assert info["waypoints_done"] == 1
    # Moved about v*dt = 3.2 m toward the waypoint. The jump to x=900 must not
    # show up as a large negative progress term.
    assert rewards[0] == pytest.approx(16.0 * env.cfg.dt, abs=0.05)
    assert rewards[0] > 0.0


def test_nlos_inflates_belief_std_more_than_open_sky():
    def std_after(obstacles):
        env = UAVFormationEnv(
            _quiet_cfg(
                obs_mode="belief",
                use_aoi=True,
                use_blockage=True,
                channel_mode="distance",
                p0=0.0,
                nlos_q_scale=4.0,
                wind_obs_noise=0.0,
                obstacles=obstacles,
                start_xy=(100.0, 400.0),
                start_noise=0.0,
                start_speed=0.0,
                max_steps=8,
            )
        )
        env.reset(seed=0)
        env.uavs[0].v = 0.0
        env.uavs[1].x, env.uavs[1].y = 300.0, 400.0
        for _ in range(5):
            env.step(np.zeros((2, env.action_dim), dtype=np.float32))
        return float(env.links[0].belief.position_std().mean())

    open_std = std_after([])
    blocked_std = std_after(
        [{"x": 200.0, "y": 400.0, "radius": 20.0, "height": 80.0, "blocks_radio": True}]
    )
    assert blocked_std > open_std * 1.5


def test_belief_point_hides_std_but_keeps_aoi():
    env = UAVFormationEnv(_quiet_cfg(obs_mode="belief_point", use_aoi=True, p0=0.0, aoi_cap=10))
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
    assert obs[1][ObsIndex.AOI] == pytest.approx(0.1)
    assert obs[1][ObsIndex.STD_X] == pytest.approx(0.0)
    assert obs[1][ObsIndex.STD_Y] == pytest.approx(0.0)
    assert obs[1][ObsIndex.STD_Z] == pytest.approx(0.0)


def test_follower_sensor_does_not_see_the_leader():
    env = UAVFormationEnv(
        EnvConfig(
            n_followers=4,
            start_noise=0.0,
            randomize_channel=False,
            randomize_wind=False,
            sensor_range=10_000.0,
        )
    )
    obs, _ = env.reset(seed=0)
    leader = env.uavs[0]
    for i, follower in enumerate(env.uavs[1:], start=1):
        true_rel = (
            (leader.x - follower.x) / env.cfg.map_width,
            (leader.y - follower.y) / env.cfg.map_height,
            (leader.z - follower.z) / env.alt_scale,
        )
        seen = (
            float(obs[i][ObsIndex.TEAMMATE_REL_X]),
            float(obs[i][ObsIndex.TEAMMATE_REL_Y]),
            float(obs[i][ObsIndex.TEAMMATE_REL_Z]),
        )
        assert obs[i][ObsIndex.TEAMMATE_PRESENT] == pytest.approx(1.0)
        assert not np.allclose(seen, true_rel, atol=1e-3)
    # A single follower has no other follower, so the teammate slot stays empty.
    alone = UAVFormationEnv(_quiet_cfg(n_followers=1))
    alone_obs, _ = alone.reset(seed=0)
    assert alone_obs[1][ObsIndex.TEAMMATE_PRESENT] == pytest.approx(0.0)


def test_slot_std_grows_with_an_unobserved_turn():
    env = UAVFormationEnv(
        _quiet_cfg(
            obs_mode="belief",
            use_aoi=True,
            p0=0.0,
            wind_obs_noise=0.0,
            start_speed=8.0,
            max_steps=12,
        )
    )
    env.reset(seed=0)
    env.uavs[0].v = 8.0
    env.uavs[0].heading = 0.0
    env.links[0].meas_heading = 0.0
    env.links[0].meas_v = 8.0
    env.reset_belief_from_stored_measurement(0)
    actions = np.zeros((2, env.action_dim), dtype=np.float32)
    actions[0, 1] = 1.0
    env.step(actions)
    early = float(np.mean(env._slot_std(0)))
    early_pos = float(env.links[0].belief.position_std().mean())
    for _ in range(5):
        env.step(actions)
    late = float(np.mean(env._slot_std(0)))
    assert late > early * 1.5
    assert late > early_pos * 2.0
    # The actor sees that growth. Other modes keep the same coast but hide it.
    belief_obs = env._observations()
    assert belief_obs[1][ObsIndex.STD_X] > 0.0
    assert belief_obs[1][ObsIndex.PRED_NLOS] == pytest.approx(0.0)
    hidden = UAVFormationEnv(_quiet_cfg(obs_mode="belief_point", use_aoi=True, p0=0.0))
    hidden.reset(seed=0)
    hidden_obs, _, _, _, _ = hidden.step(np.zeros((2, hidden.action_dim), dtype=np.float32))
    assert hidden_obs[1][ObsIndex.STD_X] == pytest.approx(0.0)
    assert hidden_obs[1][ObsIndex.PRED_NLOS] == pytest.approx(0.0)


def test_follower_heading_reward_uses_the_visible_heading():
    env = UAVFormationEnv(
        _quiet_cfg(
            obs_mode="hold",
            p0=0.0,
            w_heading=1.0,
            w_progress=0.0,
            w_formation=0.0,
            w_form_progress=0.0,
            w_smooth=0.0,
            w_step=0.0,
            w_clearance=0.0,
            formation_bonus=0.0,
            wp_bonus=0.0,
            collision_penalty=0.0,
            oob_penalty=0.0,
        )
    )
    env.reset(seed=0)
    env.uavs[0].v = 0.0
    env.uavs[0].heading = 0.0
    env.uavs[1].v = 0.0
    env.uavs[1].heading = math.pi
    env.links[0].meas_heading = math.pi
    env.links[0].meas_v = 0.0
    env.reset_belief_from_stored_measurement(0)
    _, rewards, _, _, _ = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
    # Visible heading matches the follower. The true leader heading is opposite,
    # so a privileged heading term would have been -1.
    assert rewards[1] == pytest.approx(1.0, abs=1e-5)


def test_formation_cost_caps_and_collision_is_the_worse_outcome():
    env = UAVFormationEnv(
        _quiet_cfg(
            w_formation=1.0,
            w_form_progress=0.0,
            w_heading=0.0,
            w_smooth=0.0,
            w_step=0.0,
            w_clearance=0.0,
            formation_bonus=0.0,
            formation_error_cap=80.0,
            obstacles=[],
        )
    )
    env.reset(seed=0)
    env._prev_form_err = np.array([500.0])
    env.uavs[1].x = env.uavs[0].x - 500.0
    env.uavs[1].y = env.uavs[0].y
    env.uavs[1].z = env.uavs[0].z
    rewards = env._rewards(
        actions=np.zeros((2, env.action_dim)),
        errors=np.array([500.0]),
        progress=0.0,
        captured=0,
        hit_obstacle=False,
        hit_uav=False,
        oob=np.zeros(2, dtype=bool),
    )
    assert rewards[1] == pytest.approx(-80.0)
    # Discounted cost of sitting on the cap for a full episode at the default
    # formation weight is about 159. Crashing has to cost more than that.
    default = EnvConfig()
    discount = (1.0 - 0.99 ** default.max_steps) / 0.01
    assert default.collision_penalty > default.w_formation * default.formation_error_cap * discount


def test_altitude_error_is_the_price_of_climbing_and_splits_by_roof():
    env = UAVFormationEnv(_quiet_cfg(n_followers=2))
    env.reset(seed=0)
    env.uavs[1].z = env.uavs[0].z + 30.0
    env.uavs[2].z = env.uavs[0].z
    env.links[0].blocked = 1
    env.links[1].blocked = 0
    env._accumulate_visibility(env._formation_errors())
    assert env._mean_or_nan(env._nlos_alt_sum, env._nlos_link_steps) == pytest.approx(30.0)
    assert env._mean_or_nan(env._los_alt_sum, env._los_link_steps) == pytest.approx(0.0)


def test_bad_action_shape_and_unknown_config_key():
    env = UAVFormationEnv(_quiet_cfg())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros((1, env.action_dim)))
    with pytest.raises(KeyError):
        from envs.formation_env import env_config_from_dict

        env_config_from_dict({"not_a_real_key": 1})
