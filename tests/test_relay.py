"""Relay airtime, team age, shared reward, and canyon lanes."""

from __future__ import annotations

import numpy as np

from envs.formation_env import EnvConfig, ObsIndex, UAVFormationEnv
from envs.scenario import CANYON_MAX_OBSTACLES, sample_canyon
from tests.test_env import _quiet_cfg


def _open(**kwargs) -> UAVFormationEnv:
    base = dict(
        n_followers=2,
        formation_offsets=[(-30.0, 0.0, 0.0), (-30.0, 40.0, 0.0)],
        channel_mode="perfect",
        obs_mode="belief",
        use_aoi=True,
        p0=1.0,
        leader_mode="route",
        follower_mode="slot",
        end_on_collision=False,
    )
    base.update(kwargs)
    return UAVFormationEnv(_quiet_cfg(**base))


def test_aoi_gate_gives_a_fresh_slot_back_to_the_leader():
    env = _open(relay_mode="tdma", relay_period=2, relay_aoi_gate=100)
    env.reset(seed=0)
    zeros = np.zeros((env.n_agents, env.action_dim))
    env.step(zeros)
    env.step(zeros)
    assert env.links[0].aoi == 0 and env.links[1].aoi == 0
    assert env._relay_attempts == 0
    forced = _open(relay_mode="tdma", relay_period=2, relay_aoi_gate=0)
    forced.reset(seed=0)
    forced.step(zeros)
    forced.step(zeros)
    assert forced.links[0].aoi == 1
    assert forced._packets_attempted == forced.n_followers
    env = _open(relay_mode="tdma", relay_period=2)
    env.reset(seed=0)
    zeros = np.zeros((env.n_agents, env.action_dim))
    env.step(zeros)
    assert env.links[0].aoi == 0 and env.links[1].aoi == 0
    env.links[1].aoi = 9
    env.links[1].belief.mean[0] += 500.0
    env.step(zeros)
    # Leader was silent; the relay is in flight for one actual control step.
    assert env.links[0].aoi == 1
    assert env.links[1].aoi == 10
    assert env._relay_ok == 0
    assert env._relay_tx_count == 1
    assert len(env._relay_in_flight) == 1
    # A new direct measurement next tick supersedes the delayed relay.
    env.step(zeros)
    assert env.links[1].aoi == 0
    assert env._relay_decoded == 1
    assert env._relay_ok == 0


def test_free_relay_does_not_take_the_leader_slot():
    env = _open(relay_mode="free", channel_mode="independent", p0=0.0)
    env.reset(seed=0)
    env.links[1].aoi = 9
    env.step(np.zeros((env.n_agents, env.action_dim)))
    # The leader was offered a packet and missed it, and a relay was still sent.
    assert env._packets_attempted == env.n_followers
    assert env.links[0].aoi == 1
    assert env._relay_attempts >= 1


def test_beacon_does_not_take_the_leader_slot():
    env = _open(relay_mode="beacon", relay_period=1)
    env.reset(seed=0)
    env.step(np.zeros((env.n_agents, env.action_dim)))
    # Perfect downlink still delivered, on the same step a beacon could have run.
    assert env._packets_attempted == env.n_followers
    assert env.links[0].aoi == 0
    env.links[1].aoi = 6
    env.links[1].belief.mean[0] += 400.0
    env._relay_beliefs(single=True)
    assert env.links[1].aoi == 6  # queued, not delivered in the sending call
    env._deliver_relay_packets(env._step_count + 2)
    assert env.links[1].aoi == 1
    env = _open(oracle_team_aoi=True)
    env.reset(seed=0)
    env.links[0].aoi = 8
    env.links[1].aoi = 3
    obs = env._observations()
    assert obs[1][ObsIndex.TEAM_AOI] == 3 / env.cfg.aoi_cap
    assert obs[2][ObsIndex.TEAM_AOI] == 3 / env.cfg.aoi_cap
    assert obs[0][ObsIndex.TEAM_AOI] == 0.0


def test_shared_reward_uses_the_mean_error():
    env = UAVFormationEnv(
        _quiet_cfg(
            n_followers=2,
            shared_formation_reward=True,
            w_formation=1.0,
            w_form_progress=0.0,
            w_heading=0.0,
            w_smooth=0.0,
            w_step=0.0,
            formation_bonus=0.0,
            collision_step_penalty=0.0,
            collision_penalty=0.0,
        )
    )
    env.reset(seed=0)
    shared = env._rewards(
        np.zeros((3, 3)),
        np.array([10.0, 30.0]),
        0.0,
        0,
        False,
        False,
        np.zeros(3, dtype=bool),
        None,
    )
    assert shared[1] == shared[2]
    assert abs(float(shared[1]) - (-20.0)) < 1e-4


def test_canyon_lanes_stay_clear_of_roofs():
    rng = np.random.default_rng(0)
    for _ in range(25):
        layout = sample_canyon(rng)
        street = abs(layout["formation_offsets"][0][1])
        lanes = [400.0 + k * street for k in (-2, -1, 0, 1, 2)]
        assert len(layout["obstacles"]) <= CANYON_MAX_OBSTACLES
        for obstacle in layout["obstacles"]:
            gap = min(abs(obstacle["y"] - y) - obstacle["radius"] for y in lanes)
            assert gap >= 8.0


def test_brake_when_the_climb_cannot_finish():
    env = UAVFormationEnv(
        _quiet_cfg(
            obstacles=[{"x": 180.0, "y": 400.0, "radius": 14.0, "height": 70.0, "blocks_radio": True}]
        )
    )
    env.reset(seed=0)
    uav = env.uavs[0]
    uav.x, uav.y, uav.z, uav.v, uav.heading, uav.vz = 150.0, 400.0, 40.0, 12.0, 0.0, 0.0
    _x, _y, _z, brake = env._avoid(uav, 180.0, 400.0, 40.0)
    assert brake
    uav.x = 140.0
    far = UAVFormationEnv(_quiet_cfg(obstacles=[]))
    far.reset(seed=0)
    _x, _y, _z, brake = far._avoid(far.uavs[0], 300.0, 400.0, 40.0)
    assert not brake
