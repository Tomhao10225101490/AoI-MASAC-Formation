"""Train one of the off-policy algorithms.

Examples
--------
    python train.py --config configs/smoke.yaml
    python train.py --config configs/default.yaml --seeds 0,1,2
    python train.py --config configs/vanilla_masac.yaml --episodes 30 --device cpu

The merged config is copied into the run directory. Checkpoints store that
same config so evaluation can rebuild the environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.factory import EXPECTED_OBS_MODE, build_algorithm
from envs.formation_env import UAVFormationEnv, env_config_from_dict
from utils.config import load_config, resolve_device
from utils.logger import CsvLogger, TrainLogger
from utils.rollout import evaluate_agent
from utils.seeding import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AoI-MASAC or a baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algo", default=None, help="Override the algorithm name.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, e.g. 0,1,2.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--start-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--obs-mode", default=None, choices=["aoi", "perfect", "hold", "extrapolate"])
    parser.add_argument("--channel-mode", default=None, choices=["distance", "independent", "perfect"])
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--randomize-channel", type=int, default=None, choices=[0, 1])
    parser.add_argument("--use-aoi", type=int, default=None, choices=[0, 1])
    parser.add_argument("--n-followers", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    return parser.parse_args(argv)


def apply_cli(cfg: dict, args: argparse.Namespace) -> dict:
    if args.algo:
        cfg["algo"] = args.algo
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.episodes is not None:
        cfg["train"]["episodes"] = int(args.episodes)
    if args.warmup_steps is not None:
        cfg["train"]["warmup_steps"] = int(args.warmup_steps)
    if args.start_steps is not None:
        cfg["train"]["start_steps"] = int(args.start_steps)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.device:
        cfg["device"] = args.device
    if args.log_dir:
        cfg["train"]["log_dir"] = args.log_dir
    if args.max_steps is not None:
        cfg["env"]["max_steps"] = int(args.max_steps)
    if args.obs_mode:
        cfg["env"]["obs_mode"] = args.obs_mode
    if args.channel_mode:
        cfg["env"]["channel_mode"] = args.channel_mode
    if args.p0 is not None:
        cfg["env"]["p0"] = float(args.p0)
    if args.randomize_channel is not None:
        cfg["env"]["randomize_channel"] = bool(args.randomize_channel)
    if args.use_aoi is not None:
        cfg["env"]["use_aoi"] = bool(args.use_aoi)
    if args.n_followers is not None:
        cfg["env"]["n_followers"] = int(args.n_followers)
    if args.hidden is not None:
        cfg["train"]["hidden"] = int(args.hidden)
    return cfg


def _probe_options(env_cfg: dict) -> dict:
    """Fixed channel used for the in-training evaluation probe.

    Training may randomize p0 every episode. The probe turns randomization
    off and uses the middle of the range so successive evals are comparable.
    """
    mode = env_cfg["channel_mode"]
    if env_cfg.get("randomize_channel") and mode != "perfect":
        p0 = 0.5 * (float(env_cfg["p0_low"]) + float(env_cfg["p0_high"]))
    else:
        p0 = float(env_cfg["p0"])
    return {
        "channel_mode": mode,
        "p0": p0,
        "randomize_channel": False,
        "obs_mode": env_cfg["obs_mode"],
        "use_aoi": env_cfg["use_aoi"],
        "d0": env_cfg.get("d0", 250.0),
    }


def _warn_if_mismatch(algo_name: str, env_cfg: dict) -> None:
    expected = EXPECTED_OBS_MODE.get(algo_name)
    actual = env_cfg.get("obs_mode")
    if expected is not None and actual != expected:
        print(
            f"[warn] algo={algo_name} usually uses obs_mode={expected}, "
            f"but this config has obs_mode={actual}. Training will follow the config."
        )
    if algo_name == "aoi_masac" and not env_cfg.get("use_aoi", True):
        print("[info] use_aoi is false: the AoI feature is zeroed (no-AoI ablation).")


def run_training(cfg: dict) -> Path:
    """Train and return the run directory."""
    cfg = copy.deepcopy(cfg)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device_name = resolve_device(cfg.get("device", "auto"))
    device = torch.device(device_name)
    algo_name = str(cfg.get("algo", "aoi_masac"))
    _warn_if_mismatch(algo_name, cfg["env"])

    env = UAVFormationEnv(env_config_from_dict(cfg["env"]))
    eval_env = UAVFormationEnv(env_config_from_dict(cfg["env"]))
    train_cfg = cfg["train"]
    algo = build_algorithm(
        algo_name,
        obs_dims=env.obs_dims,
        global_dim=env.global_dim,
        action_dim=env.action_dim,
        n_agents=env.n_agents,
        train_cfg=train_cfg,
        device=device,
        seed=seed,
    )

    root = Path(cfg.get("_project_root", ROOT))
    log_root = Path(train_cfg.get("log_dir", "runs"))
    if not log_root.is_absolute():
        log_root = root / log_root
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = log_root / f"{algo_name}_seed{seed}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg["device"] = device_name
    cfg["seed"] = seed
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)

    logger = TrainLogger(run_dir)
    eval_logger = CsvLogger(
        run_dir / "eval_metrics.csv",
        [
            "episode",
            "p0",
            "success_rate",
            "collision_rate",
            "formation_keeping_rate",
            "mean_formation_error",
            "mean_aoi",
            "packet_success_rate",
            "mean_return",
        ],
    )
    episodes = int(train_cfg["episodes"])
    warmup = int(train_cfg.get("warmup_steps", 1000))
    start_steps = int(train_cfg.get("start_steps", warmup))
    updates_per_step = int(train_cfg.get("updates_per_step", 1))
    eval_every = int(train_cfg.get("eval_every", 10))
    eval_episodes = int(train_cfg.get("eval_episodes", 5))
    save_every = int(train_cfg.get("save_every", 10))
    log_update_every = int(train_cfg.get("log_update_every", 50))
    rng = np.random.default_rng(seed + 12345)
    probe = _probe_options(cfg["env"])

    print(
        f"开始训练 algo={algo_name} seed={seed} device={device_name} "
        f"agents={env.n_agents} obs_dim={env.obs_dim} global_dim={env.global_dim} "
        f"obs_mode={cfg['env']['obs_mode']} channel={cfg['env']['channel_mode']}"
    )
    print(f"日志目录: {run_dir}")

    total_steps = 0
    total_updates = 0
    total_update_time = 0.0
    total_wall = 0.0
    episode_walls: list[float] = []

    for ep in range(episodes):
        ep_t0 = time.perf_counter()
        update_time = 0.0
        n_updates = 0
        loss_acc: list[dict] = []
        obs, _info = env.reset(seed=seed * 100_000 + ep)
        algo.reset_noise()
        global_state = env.global_state()
        reward_sum = np.zeros(env.n_agents, dtype=np.float64)
        terminated = False
        truncated = False
        info: dict = {}
        while not (terminated or truncated):
            if total_steps < start_steps:
                actions = rng.uniform(-1.0, 1.0, size=(env.n_agents, env.action_dim)).astype(np.float32)
            else:
                actions = algo.act(obs, explore=True, episode=ep)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_global = env.global_state()
            # Bootstrap on timeout (truncated) but not on success/collision.
            algo.push(obs, global_state, actions, rewards, next_obs, next_global, bool(terminated))
            reward_sum += rewards
            obs = next_obs
            global_state = next_global
            total_steps += 1
            if total_steps >= warmup and algo.buffer.size >= algo.batch_size:
                for _ in range(updates_per_step):
                    u0 = time.perf_counter()
                    stats = algo.update()
                    update_time += time.perf_counter() - u0
                    if stats:
                        n_updates += 1
                        total_updates += 1
                        loss_acc.append(stats)
                        if log_update_every > 0 and total_steps % log_update_every == 0:
                            logger.log_update(total_steps, stats)
        wall = time.perf_counter() - ep_t0
        total_wall += wall
        total_update_time += update_time
        episode_walls.append(wall)
        episode = dict(info["episode"])
        mean_stats = {}
        if loss_acc:
            for key in loss_acc[0]:
                mean_stats[key] = float(np.mean([row[key] for row in loss_acc]))
        row = {
            "episode": ep,
            "seed": seed,
            "algo": algo_name,
            "steps": episode["steps"],
            "total_steps": total_steps,
            "wall_time_sec": wall,
            "update_time_sec": update_time,
            "n_updates": n_updates,
            "return_sum": float(reward_sum.sum()),
            "return_mean": float(reward_sum.mean()),
            "success": episode["success"],
            "collision": episode["collision"],
            "obstacle_collision": episode["obstacle_collision"],
            "inter_uav_collision": episode["inter_uav_collision"],
            "timeout": episode["timeout"],
            "formation_keeping_rate": episode["formation_keeping_rate"],
            "mean_formation_error": episode["mean_formation_error"],
            "mean_aoi": episode["mean_aoi"],
            "max_aoi": episode["max_aoi"],
            "packet_success_rate": episode["packet_success_rate"],
            "mean_outage_length": episode["mean_outage_length"],
            "waypoint_fraction": episode["waypoint_fraction"],
            "nlos_fraction": episode["nlos_fraction"],
            "min_clearance": episode["min_clearance"],
            "mean_belief_std": episode["mean_belief_std"],
            "los_mean_formation_error": episode["los_mean_formation_error"],
            "nlos_mean_formation_error": episode["nlos_mean_formation_error"],
            "los_mean_aoi": episode["los_mean_aoi"],
            "nlos_mean_aoi": episode["nlos_mean_aoi"],
            "los_mean_slot_std": episode["los_mean_slot_std"],
            "nlos_mean_slot_std": episode["nlos_mean_slot_std"],
            "p0": episode["p0"],
            "critic_loss": mean_stats.get("critic_loss", ""),
            "actor_loss": mean_stats.get("actor_loss", ""),
            "alpha": mean_stats.get("alpha", ""),
            "q": mean_stats.get("q", ""),
        }
        logger.log_episode(row)
        print(
            f"[回合 {ep + 1}/{episodes}] 步数={episode['steps']} 回报={row['return_sum']:.2f} "
            f"成功={episode['success']:.0f} 碰撞={episode['collision']:.0f} "
            f"编队保持={episode['formation_keeping_rate']:.2f} 平均AoI={episode['mean_aoi']:.2f} "
            f"p0={episode['p0']:.2f} 用时={wall:.2f}s"
        )
        if eval_every > 0 and (ep + 1) % eval_every == 0:
            rows = evaluate_agent(
                eval_env,
                lambda observation: algo.act(observation, explore=False),
                n_episodes=eval_episodes,
                base_seed=10_000 + seed * 1000 + ep,
                options=probe,
            )
            from metrics import summarize_episodes

            summary = summarize_episodes(rows)
            eval_logger.log(
                {
                    "episode": ep + 1,
                    "p0": probe["p0"],
                    "success_rate": summary["success_rate"],
                    "collision_rate": summary["collision_rate"],
                    "formation_keeping_rate": summary["formation_keeping_rate"],
                    "mean_formation_error": summary["mean_formation_error"],
                    "mean_aoi": summary["mean_aoi"],
                    "packet_success_rate": summary["packet_success_rate"],
                    "mean_return": summary["mean_return"],
                }
            )
            print(
                f"  评估@{probe['channel_mode']} p0={probe['p0']:.2f}: "
                f"成功={summary['success_rate']:.2f} 碰撞={summary['collision_rate']:.2f} "
                f"编队保持={summary['formation_keeping_rate']:.2f} 平均AoI={summary['mean_aoi']:.2f}"
            )
        if save_every > 0 and (ep + 1) % save_every == 0:
            _save_checkpoint(run_dir / f"ckpt_ep{ep + 1:04d}.pt", algo, cfg, algo_name, ep + 1, total_steps, env)

    _save_checkpoint(run_dir / "ckpt_final.pt", algo, cfg, algo_name, episodes, total_steps, env)
    speed = {
        "label": "short-or-configured-run",
        "device": device_name,
        "cpu_count": __import__("os").cpu_count(),
        "algo": algo_name,
        "seed": seed,
        "episodes": episodes,
        "total_steps": total_steps,
        "total_updates": total_updates,
        "mean_episode_sec": float(np.mean(episode_walls)) if episode_walls else None,
        "median_episode_sec": float(np.median(episode_walls)) if episode_walls else None,
        "mean_step_sec": (total_wall / total_steps) if total_steps else None,
        "mean_update_sec": (total_update_time / total_updates) if total_updates else None,
        "total_wall_sec": total_wall,
        "note": "Wall time includes environment steps and gradient updates. This is a measured run on this machine, not a paper result.",
    }
    with (run_dir / "speed.json").open("w", encoding="utf-8") as handle:
        json.dump(speed, handle, indent=2)
    print(
        f"训练结束。平均每回合 {speed['mean_episode_sec']:.3f}s，"
        f"平均每步 {speed['mean_step_sec']:.4f}s，"
        f"平均每次更新 {speed['mean_update_sec']:.4f}s。"
        if speed["mean_update_sec"] is not None
        else f"训练结束。平均每回合 {speed['mean_episode_sec']:.3f}s（没有发生参数更新）。"
    )
    logger.close()
    eval_logger.close()
    return run_dir


def _save_checkpoint(path: Path, algo, cfg: dict, algo_name: str, episode: int, total_steps: int, env) -> None:
    torch.save(
        {
            "algo": algo.state_dict(),
            "config": cfg,
            "algo_name": algo_name,
            "episode": episode,
            "total_steps": total_steps,
            "obs_dim": env.obs_dim,
            "global_dim": env.global_dim,
            "n_agents": env.n_agents,
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    apply_cli(cfg, args)
    if args.seeds:
        seeds = [int(part) for part in args.seeds.split(",") if part.strip() != ""]
    else:
        seeds = [int(cfg.get("seed", 0))]
    for seed in seeds:
        one = copy.deepcopy(cfg)
        one["seed"] = seed
        run_training(one)


if __name__ == "__main__":
    main()
