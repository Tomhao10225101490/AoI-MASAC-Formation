"""Evaluate a checkpoint under a chosen channel.

The observation mode defaults to ``auto``:

* A policy trained with ``obs_mode=perfect`` is deployed on a lossy link by
  switching the observation builder to ``hold``. Otherwise the network would
  still be handed the true leader state and the lossy link would not matter.
* Every other policy keeps the observation mode it was trained with.

Pass ``--obs-mode perfect`` explicitly if you want the oracle leader state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.factory import build_algorithm
from envs.formation_env import UAVFormationEnv, env_config_from_dict
from metrics import summarize_episodes
from utils.config import resolve_device
from utils.rollout import collect_episode, evaluate_agent
from utils.seeding import set_seed


def resolve_test_observation(
    trained_obs_mode: str,
    trained_use_aoi: bool,
    channel_mode: str,
    p0: float,
    requested_obs: str | None,
    requested_use_aoi: bool | None,
) -> tuple[str, bool, str]:
    """Pick the observation builder used at test time."""
    if requested_obs in (None, "auto"):
        lossy = channel_mode != "perfect" and p0 < 1.0 - 1e-12
        if trained_obs_mode == "perfect" and lossy:
            return (
                "hold",
                False,
                "perfect-trained policy on a lossy link: obs_mode switched to hold "
                "(stale leader state, no AoI feature). Pass --obs-mode perfect for an oracle.",
            )
        return trained_obs_mode, bool(trained_use_aoi), ""
    obs_mode = requested_obs
    if requested_use_aoi is None:
        use_aoi = bool(trained_use_aoi) if obs_mode == "aoi" else False
    else:
        use_aoi = requested_use_aoi
    return obs_mode, use_aoi, ""


def load_trained(checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    env = UAVFormationEnv(env_config_from_dict(cfg["env"]))
    if int(ckpt.get("obs_dim", env.obs_dim)) != env.obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim {ckpt.get('obs_dim')} != env obs_dim {env.obs_dim}. "
            "The saved config does not match this code."
        )
    algo_name = ckpt.get("algo_name", cfg.get("algo", "aoi_masac"))
    algo = build_algorithm(
        algo_name,
        obs_dims=env.obs_dims,
        global_dim=env.global_dim,
        action_dim=env.action_dim,
        n_agents=env.n_agents,
        train_cfg=cfg["train"],
        device=device,
        seed=int(cfg.get("seed", 0)),
    )
    algo.load_state_dict(ckpt["algo"])
    return ckpt, env, algo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained formation policy.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--channel-mode", default=None, choices=["distance", "independent", "perfect"])
    parser.add_argument("--p0", type=float, default=None)
    parser.add_argument("--d0", type=float, default=None)
    parser.add_argument(
        "--obs-mode",
        default="auto",
        choices=["auto", "aoi", "perfect", "hold", "extrapolate"],
    )
    parser.add_argument("--use-aoi", type=int, default=None, choices=[0, 1])
    parser.add_argument("--randomize-channel", type=int, default=0, choices=[0, 1])
    parser.add_argument("--save-traj", default=None, help="Path to write the first episode as .npz")
    parser.add_argument("--out", default=None, help="Directory for episodes.csv and summary.json")
    return parser.parse_args(argv)


def build_options(args: argparse.Namespace, cfg: dict) -> tuple[dict, str]:
    env_cfg = cfg["env"]
    channel_mode = args.channel_mode or (
        "independent" if env_cfg["channel_mode"] == "perfect" else env_cfg["channel_mode"]
    )
    # A perfect-trained policy has p0=1 in its config. If the caller did not
    # pass --p0, keep the trained p0 when the channel mode is unchanged, and
    # use 0.7 as a mild default loss when they explicitly switch onto a lossy
    # independent/distance channel from a perfect trainer.
    if args.p0 is not None:
        p0 = float(args.p0)
    elif args.channel_mode is not None and env_cfg["channel_mode"] == "perfect" and channel_mode != "perfect":
        p0 = 0.7
    else:
        p0 = float(env_cfg["p0"])
    requested_use = None if args.use_aoi is None else bool(args.use_aoi)
    obs_mode, use_aoi, note = resolve_test_observation(
        trained_obs_mode=env_cfg["obs_mode"],
        trained_use_aoi=bool(env_cfg.get("use_aoi", True)),
        channel_mode=channel_mode,
        p0=p0,
        requested_obs=args.obs_mode,
        requested_use_aoi=requested_use,
    )
    options = {
        "channel_mode": channel_mode,
        "p0": p0,
        "randomize_channel": bool(args.randomize_channel),
        "obs_mode": obs_mode,
        "use_aoi": use_aoi,
        "d0": float(args.d0) if args.d0 is not None else float(env_cfg.get("d0", 250.0)),
    }
    return options, note


def run_evaluation(args: argparse.Namespace) -> dict:
    device = torch.device(resolve_device(args.device))
    set_seed(args.seed)
    ckpt, env, algo = load_trained(args.checkpoint, device)
    options, note = build_options(args, ckpt["config"])
    if note:
        print("[info] " + note)
    print(
        f"评估 {ckpt.get('algo_name')}  回合={args.episodes}  "
        f"channel={options['channel_mode']} p0={options['p0']:.3f}  "
        f"obs={options['obs_mode']} use_aoi={options['use_aoi']}"
    )

    def act(observation):
        return algo.act(observation, explore=False)

    rows = []
    trajectory = None
    for i in range(args.episodes):
        record = args.save_traj is not None and i == 0
        episode, traj = collect_episode(
            env,
            act,
            reset_seed=args.seed + i,
            options=options,
            record=record,
        )
        rows.append(episode)
        if traj is not None:
            trajectory = traj
    summary = summarize_episodes(rows)
    summary.update(
        {
            "channel_mode": options["channel_mode"],
            "p0": options["p0"],
            "obs_mode": options["obs_mode"],
            "use_aoi": float(options["use_aoi"]),
            "algo": ckpt.get("algo_name", ""),
            "checkpoint": str(args.checkpoint),
        }
    )
    _print_summary(summary)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        _write_episodes(out / "episodes.csv", rows)
        with (out / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"结果已写入 {out}")
    if trajectory is not None:
        traj_path = Path(args.save_traj)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(traj_path, **trajectory)
        print(f"轨迹已写入 {traj_path}")
    return summary


def _print_summary(summary: dict) -> None:
    labels = [
        ("success_rate", "成功率"),
        ("collision_rate", "碰撞率"),
        ("obstacle_collision_rate", "障碍碰撞率"),
        ("inter_uav_collision_rate", "机间碰撞率"),
        ("timeout_rate", "超时率"),
        ("formation_keeping_rate", "编队保持率"),
        ("mean_formation_error", "平均编队误差(米)"),
        ("mean_aoi", "平均AoI(步)"),
        ("packet_success_rate", "链路成功率"),
        ("mean_outage_length", "平均中断长度(步)"),
        ("mean_return", "平均总回报"),
        ("mean_steps", "平均步数"),
        ("waypoint_fraction", "航路点完成比例"),
        ("nlos_fraction", "视线被建筑物挡住的比例"),
        ("min_clearance", "最近障碍间隙(米)"),
    ]
    print(f"共 {int(summary['n_episodes'])} 个回合（下面是平均值 ± 样本标准差）")
    for key, label in labels:
        mean = summary[key]
        std = summary[key + "_std"]
        print(f"  {label:20s} {mean:8.3f}  ± {std:.3f}")


def _write_episodes(path: Path, rows: list[dict]) -> None:
    import csv

    keys = [
        "success",
        "collision",
        "obstacle_collision",
        "inter_uav_collision",
        "timeout",
        "formation_keeping_rate",
        "mean_formation_error",
        "mean_aoi",
        "max_aoi",
        "packet_success_rate",
        "mean_outage_length",
        "steps",
        "p0",
        "return_sum",
        "return_mean",
        "waypoint_fraction",
        "nlos_fraction",
        "min_clearance",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in keys})


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_evaluation(args)


if __name__ == "__main__":
    main()
