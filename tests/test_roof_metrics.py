"""Roof class is part of the paper's claim, not a plotting convenience.

An overflyable roof (height at or below the ceiling) and a building above
the ceiling are opposite decisions. The non-line-of-sight average mixes them.
"""

from __future__ import annotations

import math

import numpy as np

from envs.formation_env import EnvConfig, UAVFormationEnv
from metrics import summarize_episodes


def _env() -> UAVFormationEnv:
    env = UAVFormationEnv(
        EnvConfig(
            n_followers=1,
            max_steps=1,
            randomize_channel=False,
            randomize_wind=False,
            p0=1.0,
        )
    )
    env.reset(seed=0)
    env.wind_x = 0.0
    env.wind_y = 0.0
    for uav in env.uavs:
        uav.v = 0.0
        uav.vz = 0.0
    return env


def _place(env: UAVFormationEnv, leader, follower) -> dict:
    env.uavs[0].x, env.uavs[0].y, env.uavs[0].z = leader
    env.uavs[1].x, env.uavs[1].y, env.uavs[1].z = follower
    _, _, terminated, truncated, info = env.step(np.zeros((2, 3), dtype=np.float32))
    assert terminated or truncated
    return info["episode"]


def test_block_class_splits_the_two_roofs_on_the_route():
    env = _env()
    # 50 m roof at (400, 490), radius 36. Cruise altitude cuts it.
    assert env._block_class(360, 490, 40, 440, 490, 40) == "overfly"
    # 160 m building at (660, 440), above the 120 m ceiling.
    assert env._block_class(620, 440, 40, 700, 440, 40) == "impassable"
    assert env._block_class(100, 100, 40, 130, 100, 40) == "los"
    # One tall roof on the segment is enough. The short roof does not cancel it.
    assert env._block_class(360, 490, 40, 700, 440, 40) == "impassable"


def test_overfly_and_impassable_samples_partition_blocked_links():
    over = _place(_env(), (360.0, 490.0, 40.0), (440.0, 490.0, 40.0))
    assert over["overfly_link_steps"] == 1.0
    assert over["impassable_link_steps"] == 0.0
    assert over["overfly_link_steps"] + over["impassable_link_steps"] == 1.0
    assert math.isfinite(over["overfly_mean_z"])
    assert math.isnan(over["impassable_mean_z"])

    blocked = _place(_env(), (620.0, 440.0, 40.0), (700.0, 440.0, 40.0))
    assert blocked["impassable_link_steps"] == 1.0
    assert blocked["overfly_link_steps"] == 0.0
    assert math.isfinite(blocked["impassable_mean_leader_z"])


def test_clear_rate_is_one_only_above_the_short_roof():
    # Follower at the 50 m roof's center. Fuselage radius is 4 m, so 60 m clears
    # and 40 m does not. The leader stays outside the footprint.
    cleared = _place(_env(), (250.0, 490.0, 40.0), (400.0, 490.0, 60.0))
    assert cleared["overfly_inside_steps"] == 1.0
    assert cleared["overfly_clear_rate"] == 1.0

    low = _place(_env(), (250.0, 490.0, 40.0), (400.0, 490.0, 40.0))
    assert low["overfly_inside_steps"] == 1.0
    assert low["overfly_clear_rate"] == 0.0


def test_summary_keeps_a_missing_roof_class_as_nan():
    row = {
        "success": 0.0,
        "collision": 0.0,
        "obstacle_collision": 0.0,
        "inter_uav_collision": 0.0,
        "timeout": 1.0,
        "formation_keeping_rate": 0.0,
        "mean_formation_error": 10.0,
        "mean_aoi": 1.0,
        "max_aoi": 2.0,
        "packet_success_rate": 0.8,
        "mean_outage_length": 0.0,
        "steps": 10,
        "p0": 0.6,
        "return_sum": 0.0,
        "return_mean": 0.0,
        "waypoint_fraction": 0.0,
        "nlos_fraction": 0.0,
        "min_clearance": 5.0,
        "mean_belief_std": 1.0,
        "los_mean_formation_error": 8.0,
        "nlos_mean_formation_error": float("nan"),
        "los_mean_aoi": 0.5,
        "nlos_mean_aoi": float("nan"),
        "los_mean_slot_std": 1.0,
        "nlos_mean_slot_std": float("nan"),
        "los_mean_alt_error": 1.0,
        "nlos_mean_alt_error": float("nan"),
        "los_mean_z": 40.0,
        "overfly_mean_alt_error": float("nan"),
        "impassable_mean_alt_error": float("nan"),
        "overfly_mean_z": float("nan"),
        "impassable_mean_z": float("nan"),
        "overfly_mean_leader_z": float("nan"),
        "impassable_mean_leader_z": float("nan"),
        "overfly_clear_rate": float("nan"),
        "overfly_link_steps": 0.0,
        "impassable_link_steps": 0.0,
        "overfly_inside_steps": 0.0,
        "near_roof_mean_z": float("nan"),
        "near_roof_mean_alt_error": float("nan"),
        "near_roof_mean_leader_z": float("nan"),
        "near_roof_link_steps": 0.0,
        "tall_mean_z": float("nan"),
        "tall_mean_alt_error": float("nan"),
        "tall_mean_leader_z": float("nan"),
        "tall_link_steps": 0.0,
        "far_roof_mean_z": float("nan"),
        "far_roof_mean_alt_error": float("nan"),
        "far_roof_mean_leader_z": float("nan"),
        "far_roof_link_steps": 0.0,
    }
    summary = summarize_episodes([row])
    assert summary["n_episodes_with_overfly"] == 0.0
    assert summary["n_episodes_with_impassable"] == 0.0
    assert math.isnan(summary["overfly_mean_z"])
    assert summary["overfly_link_steps"] == 0.0


def test_zero_residual_tracks_the_believed_slot():
    """A zero command must leave the slot tracker in charge."""
    env = UAVFormationEnv(
        EnvConfig(
            n_followers=2,
            leader_mode="route",
            follower_mode="residual",
            end_on_collision=False,
            max_steps=400,
            randomize_channel=False,
            randomize_wind=False,
            p0=1.0,
        )
    )
    env.reset(seed=2)
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros((env.n_agents, 3), dtype=np.float32))
    episode = info["episode"]
    assert episode["mean_formation_error"] < 80.0
    assert episode["waypoint_fraction"] >= 0.75


def test_route_leader_and_slot_followers_cross_the_three_acts():
    """The measurement is only interpretable if the route actually passes the roofs."""
    env = UAVFormationEnv(
        EnvConfig(
            n_followers=2,
            leader_mode="route",
            follower_mode="slot",
            end_on_collision=False,
            max_steps=900,
            randomize_channel=False,
            randomize_wind=False,
            p0=1.0,
        )
    )
    env.reset(seed=1)
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros((env.n_agents, 3), dtype=np.float32))
    episode = info["episode"]
    assert episode["waypoint_fraction"] >= 0.75
    assert episode["near_roof_link_steps"] > 0
    assert episode["tall_link_steps"] > 0
    assert episode["far_roof_link_steps"] > 0
    # Passing the short roofs, the route climbs. Passing the 160 m building, it stays low.
    assert episode["near_roof_mean_leader_z"] > 70.0
    assert episode["far_roof_mean_leader_z"] > 60.0
    assert episode["tall_mean_leader_z"] < 55.0
