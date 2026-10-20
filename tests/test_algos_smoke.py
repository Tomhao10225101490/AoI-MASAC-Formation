"""A few hundred environment steps of every trainable algorithm, on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from algos.factory import build_algorithm
from train import run_training
from utils.config import load_config


ALGOS = [
    "aoi_masac",
    "vanilla_masac",
    "masac_hold",
    "masac_extrapolate",
    "maddpg",
    "matd3",
]

OBS_FOR_ALGO = {
    "aoi_masac": ("aoi", True, "distance", True),
    "vanilla_masac": ("perfect", False, "perfect", False),
    "masac_hold": ("hold", False, "distance", True),
    "masac_extrapolate": ("extrapolate", False, "distance", True),
    "maddpg": ("hold", False, "distance", True),
    "matd3": ("hold", False, "distance", True),
}


def _tiny_cfg(algo: str, log_dir: Path) -> dict:
    obs_mode, use_aoi, channel_mode, randomize = OBS_FOR_ALGO[algo]
    cfg = load_config("configs/smoke.yaml")
    cfg["algo"] = algo
    cfg["seed"] = 0
    cfg["device"] = "cpu"
    cfg["env"].update(
        {
            "n_followers": 2,
            "max_steps": 50,
            "obs_mode": obs_mode,
            "use_aoi": use_aoi,
            "channel_mode": channel_mode,
            "randomize_channel": randomize,
            "p0": 1.0 if channel_mode == "perfect" else 0.7,
        }
    )
    cfg["train"].update(
        {
            "episodes": 4,          # 4 * 50 = 200 environment steps
            "warmup_steps": 40,
            "start_steps": 20,
            "batch_size": 32,
            "hidden": 32,
            "buffer_size": 2000,
            "updates_per_step": 1,
            "ou_episodes": 0,
            "eval_every": 0,
            "save_every": 0,
            "log_update_every": 1000,
            "log_dir": str(log_dir),
            "expl_noise": 0.1,
            "policy_delay": 2,
        }
    )
    return cfg


@pytest.mark.parametrize("algo", ALGOS)
def test_short_training(algo: str, tmp_path: Path):
    run_dir = run_training(_tiny_cfg(algo, tmp_path))
    ckpt_path = run_dir / "ckpt_final.pt"
    assert ckpt_path.is_file()
    metrics = (run_dir / "metrics.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics) == 5  # header + 4 episodes
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["algo_name"] == algo
    # The budget is 4 episodes x 50 steps. A collision ends the episode early,
    # so the realized step count can be lower. It still has to be a real
    # training run, not a single step.
    assert 120 <= ckpt["total_steps"] <= 4 * 50
    # Losses logged on the last episodes must be finite. The first episode
    # can still be inside warmup, so scan from the bottom.
    last = metrics[-1].split(",")
    header = metrics[0].split(",")
    critic_idx = header.index("critic_loss")
    assert last[critic_idx] not in ("",)
    assert np.isfinite(float(last[critic_idx]))


def test_mappo_is_an_explicit_stub():
    with pytest.raises(NotImplementedError, match="TODO"):
        build_algorithm(
            "mappo",
            obs_dims=[4, 4],
            global_dim=8,
            action_dim=2,
            n_agents=2,
            train_cfg={},
            device=torch.device("cpu"),
            seed=0,
        )


def test_checkpoint_roundtrip_is_deterministic(tmp_path: Path):
    run_dir = run_training(_tiny_cfg("aoi_masac", tmp_path))
    from evaluate import load_trained

    _, env, algo = load_trained(run_dir / "ckpt_final.pt", torch.device("cpu"))
    obs, _ = env.reset(seed=123)
    a = algo.act(obs, explore=False)
    b = algo.act(obs, explore=False)
    assert np.allclose(a, b)
    assert a.shape == (env.n_agents, 2)


def test_every_yaml_config_builds_an_env_and_a_network():
    from envs.formation_env import UAVFormationEnv, env_config_from_dict

    configs = sorted(Path("configs").glob("*.yaml"))
    assert configs, "configs/ is empty"
    for path in configs:
        cfg = load_config(path)
        env = UAVFormationEnv(env_config_from_dict(cfg["env"]))
        algo = build_algorithm(
            cfg["algo"],
            obs_dims=env.obs_dims,
            global_dim=env.global_dim,
            action_dim=env.action_dim,
            n_agents=env.n_agents,
            train_cfg={"hidden": 16, "buffer_size": 100, "batch_size": 8},
            device=torch.device("cpu"),
            seed=0,
        )
        obs, _ = env.reset(seed=0)
        actions = algo.act(obs, explore=False)
        assert actions.shape == (env.n_agents, env.action_dim)
