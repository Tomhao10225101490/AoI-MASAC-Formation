"""Sweep nominal packet-loss rate from 0% to 70% and write a CSV.

The x-axis is a single number, so the channel is switched to
``independent`` mode where every follower loses a packet with probability
``loss_rate`` regardless of distance (``p0 = 1 - loss_rate``). That is the
curve the paper asks for. It is not the same thing as the distance-dependent
channel used in training; the README explains how to read the two.

Distance-dependent training policies are still evaluated here. Their actor
just has to fly under a channel whose loss rate we can put on a plot axis.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import load_trained, resolve_test_observation
from metrics import summarize_episodes
from utils.config import resolve_device
from utils.rollout import evaluate_agent
from utils.seeding import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Packet-loss sweep for one checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--loss-rates",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7",
        help="Comma-separated nominal loss rates in [0, 1].",
    )
    parser.add_argument("--obs-mode", default="auto")
    parser.add_argument("--out", default="results/sweep")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(resolve_device(args.device))
    set_seed(args.seed)
    ckpt, env, algo = load_trained(args.checkpoint, device)
    trained_env = ckpt["config"]["env"]
    loss_rates = [float(part) for part in args.loss_rates.split(",") if part.strip() != ""]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"

    def act(observation):
        return algo.act(observation, explore=False)

    rows = []
    for loss in loss_rates:
        if not 0.0 <= loss <= 1.0:
            raise ValueError(f"loss rate {loss} is outside [0, 1]")
        p0 = 1.0 - loss
        obs_mode, use_aoi, note = resolve_test_observation(
            trained_obs_mode=trained_env["obs_mode"],
            trained_use_aoi=bool(trained_env.get("use_aoi", True)),
            channel_mode="independent",
            p0=p0,
            requested_obs=args.obs_mode,
            requested_use_aoi=None,
        )
        if note and loss == loss_rates[0]:
            print("[info] " + note)
        options = {
            "channel_mode": "independent",
            "p0": p0,
            "randomize_channel": False,
            "obs_mode": obs_mode,
            "use_aoi": use_aoi,
            "d0": float(trained_env.get("d0", 250.0)),
        }
        episodes = evaluate_agent(
            env,
            act,
            n_episodes=args.episodes,
            base_seed=args.seed + int(round(loss * 1000)),
            options=options,
        )
        summary = summarize_episodes(episodes)
        summary.update(
            {
                "loss_rate": loss,
                "p0": p0,
                "channel_mode": "independent",
                "obs_mode": obs_mode,
                "use_aoi": float(use_aoi),
                "algo": ckpt.get("algo_name", ""),
            }
        )
        rows.append(summary)
        print(
            f"丢包率={loss:.0%}  成功={summary['success_rate']:.3f}  "
            f"碰撞={summary['collision_rate']:.3f}  "
            f"编队保持={summary['formation_keeping_rate']:.3f}  "
            f"平均AoI={summary['mean_aoi']:.2f}  "
            f"实测链路成功={summary['packet_success_rate']:.3f}"
        )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"扫描结果已写入 {csv_path}")


if __name__ == "__main__":
    main()
