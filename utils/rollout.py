"""Evaluation rollouts shared by train.py, evaluate.py, and the sweep script."""

from __future__ import annotations

from typing import Callable

import numpy as np

from envs.formation_env import UAVFormationEnv


def collect_episode(
    env: UAVFormationEnv,
    act_fn: Callable[[list[np.ndarray]], np.ndarray],
    reset_seed: int | None = None,
    options: dict | None = None,
    record: bool = False,
) -> tuple[dict, dict | None]:
    """Run one episode with a frozen action function.

    Returns the episode metric dict (including return) and, if ``record`` is
    set, a dict of stacked trajectories.
    """
    obs, _info = env.reset(seed=reset_seed, options=options)
    reward_sum = np.zeros(env.n_agents, dtype=np.float64)
    frames: list[dict] = []
    if record:
        frames.append(env.snapshot())
    terminated = False
    truncated = False
    info: dict = {}
    while not (terminated or truncated):
        actions = np.asarray(act_fn(obs), dtype=np.float32)
        obs, rewards, terminated, truncated, info = env.step(actions)
        reward_sum += rewards
        if record:
            frames.append(env.snapshot())
    if info.get("episode") is None:
        raise RuntimeError("episode ended without an episode summary")
    episode = dict(info["episode"])
    episode["return_sum"] = float(reward_sum.sum())
    episode["return_mean"] = float(reward_sum.mean())
    episode["return_per_agent"] = reward_sum.astype(float).tolist()
    trajectory = _stack_frames(frames, env) if record else None
    return episode, trajectory


def _stack_frames(frames: list[dict], env: UAVFormationEnv) -> dict:
    return {
        "uav_xy": np.stack([f["uav_xy"] for f in frames], axis=0),
        "heading": np.stack([f["heading"] for f in frames], axis=0),
        "obstacle_xy": np.stack([f["obstacle_xy"] for f in frames], axis=0),
        "obstacle_r": frames[0]["obstacle_r"],
        "aoi": np.stack([f["aoi"] for f in frames], axis=0),
        "goal": frames[0]["goal"],
        "map_size": np.array([env.cfg.map_width, env.cfg.map_height], dtype=np.float32),
        "uav_radius": np.float32(env.cfg.uav_radius),
    }


def evaluate_agent(
    env: UAVFormationEnv,
    act_fn: Callable[[list[np.ndarray]], np.ndarray],
    n_episodes: int,
    base_seed: int,
    options: dict | None = None,
) -> list[dict]:
    rows = []
    for i in range(n_episodes):
        episode, _ = collect_episode(
            env,
            act_fn,
            reset_seed=base_seed + i,
            options=options,
            record=False,
        )
        rows.append(episode)
    return rows
