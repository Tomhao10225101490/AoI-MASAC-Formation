"""Build an algorithm object from a config name."""

from __future__ import annotations

import torch

from algos.maddpg import MADDPG
from algos.mappo import MAPPO
from algos.masac import MASAC
from algos.matd3 import MATD3

# Names that all train the MASAC class. The environment obs_mode, not the
# class, is what distinguishes them. See configs/*.yaml.
MASAC_NAMES = ("aoi_masac", "masac", "vanilla_masac", "masac_hold", "masac_extrapolate")

# Expected actor observation mode. train.py warns if a config disagrees.
EXPECTED_OBS_MODE = {
    "aoi_masac": "aoi",
    "masac": "aoi",
    "vanilla_masac": "perfect",
    "masac_hold": "hold",
    "masac_extrapolate": "extrapolate",
    "maddpg": "hold",
    "matd3": "hold",
}


def build_algorithm(
    name: str,
    obs_dims: list[int],
    global_dim: int,
    action_dim: int,
    n_agents: int,
    train_cfg: dict,
    device: torch.device,
    seed: int,
):
    """Return a trainer. MAPPO raises NotImplementedError on purpose."""
    key = name.lower().strip()
    kwargs = dict(
        obs_dims=obs_dims,
        global_dim=global_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        train_cfg=train_cfg,
        device=device,
        seed=seed,
    )
    if key in MASAC_NAMES:
        algo = MASAC(**kwargs)
        algo.algo_name = key
        return algo
    if key == "maddpg":
        return MADDPG(**kwargs)
    if key == "matd3":
        return MATD3(**kwargs)
    if key == "mappo":
        return MAPPO(**kwargs)
    known = ", ".join(sorted(set(MASAC_NAMES) | {"maddpg", "matd3", "mappo"}))
    raise ValueError(f"Unknown algorithm {name!r}. Known names: {known}")
