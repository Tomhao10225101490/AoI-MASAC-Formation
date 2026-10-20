"""Metric aggregation matches the definitions in docs/METRICS.md."""

from __future__ import annotations

from metrics import summarize_episodes


def _row(**kwargs):
    base = dict(
        success=0.0,
        collision=0.0,
        obstacle_collision=0.0,
        inter_uav_collision=0.0,
        timeout=1.0,
        formation_keeping_rate=0.5,
        mean_formation_error=10.0,
        mean_aoi=1.0,
        max_aoi=2.0,
        packet_success_rate=0.8,
        mean_outage_length=0.0,
        steps=10,
        p0=0.5,
        return_sum=1.0,
        return_mean=0.2,
        waypoint_fraction=0.25,
        nlos_fraction=0.0,
        min_clearance=10.0,
        mean_belief_std=1.0,
    )
    base.update(kwargs)
    return base


def test_summarize_macro_average_and_partition():
    rows = [
        _row(success=1, collision=0, timeout=0, formation_keeping_rate=1.0, return_sum=10),
        _row(success=0, collision=1, obstacle_collision=1, timeout=0, formation_keeping_rate=0.0, return_sum=0),
    ]
    summary = summarize_episodes(rows)
    assert summary["n_episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["obstacle_collision_rate"] == 0.5
    assert summary["timeout_rate"] == 0.0
    assert summary["formation_keeping_rate"] == 0.5
    assert summary["mean_return"] == 5.0
    assert summary["success_rate_std"] > 0.0


def test_episode_flags_sum_to_one_on_a_short_rollout():
    import numpy as np

    from envs.formation_env import EnvConfig, UAVFormationEnv

    env = UAVFormationEnv(EnvConfig(n_followers=1, max_steps=8, randomize_channel=False, p0=1.0))
    env.reset(seed=0)
    info = {}
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros((2, env.action_dim), dtype=np.float32))
    ep = info["episode"]
    assert ep["success"] + ep["collision"] + ep["timeout"] == 1.0
    summary = summarize_episodes([{**ep, "return_sum": 0.0, "return_mean": 0.0}])
    assert summary["n_episodes"] == 1.0
