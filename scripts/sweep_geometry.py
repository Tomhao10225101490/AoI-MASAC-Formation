"""Evaluate a checkpoint on the distance x blockage channel.

This is the paper's main comparison. Packet success depends on 3D range and
on whether the segment hits a building. The x-axis is the episode's p0, the
same knob training randomizes, not an independent coin-flip loss rate.

The CSV includes formation error, AoI, and slot standard deviation split
into line-of-sight and non-line-of-sight follower-steps. That split is where
the belief is supposed to matter. The independent-loss sweep remains
available as a stress test; it is not this curve.
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
    parser = argparse.ArgumentParser(description="Distance x blockage evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--p0-values",
        default="1.0,0.8,0.6,0.45,0.3",
        help="Comma-separated p0 values. Blockage and distance stay on.",
    )
    parser.add_argument("--obs-mode", default="auto")
    parser.add_argument("--out", default="results/geometry")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(resolve_device(args.device))
    set_seed(args.seed)
    ckpt, env, algo = load_trained(args.checkpoint, device)
    trained_env = ckpt["config"]["env"]
    p0_values = [float(part) for part in args.p0_values.split(",") if part.strip() != ""]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "geometry.csv"

    def act(observation):
        return algo.act(observation, explore=False)

    rows = []
    for p0 in p0_values:
        if not 0.0 <= p0 <= 1.0:
            raise ValueError(f"p0 {p0} is outside [0, 1]")
        obs_mode, use_aoi, note = resolve_test_observation(
            trained_obs_mode=trained_env["obs_mode"],
            trained_use_aoi=bool(trained_env.get("use_aoi", True)),
            channel_mode="distance",
            p0=p0,
            requested_obs=args.obs_mode,
            requested_use_aoi=None,
        )
        if note and p0 == p0_values[0]:
            print("[info] " + note)
        options = {
            "channel_mode": "distance",
            "p0": p0,
            "randomize_channel": False,
            "use_blockage": True,
            "obs_mode": obs_mode,
            "use_aoi": use_aoi,
            "d0": float(trained_env.get("d0", 200.0)),
        }
        episodes = evaluate_agent(
            env,
            act,
            n_episodes=args.episodes,
            base_seed=args.seed + int(round(p0 * 1000)),
            options=options,
        )
        summary = summarize_episodes(episodes)
        summary.update(
            {
                "p0": p0,
                "channel_mode": "distance",
                "use_blockage": 1.0,
                "obs_mode": obs_mode,
                "use_aoi": float(use_aoi),
                "algo": ckpt.get("algo_name", ""),
            }
        )
        rows.append(summary)
        print(
            f"p0={p0:.2f}  成功={summary['success_rate']:.3f}  "
            f"碰撞={summary['collision_rate']:.3f}  "
            f"视距编队误差={summary['los_mean_formation_error']:.2f}  "
            f"非视距编队误差={summary['nlos_mean_formation_error']:.2f}  "
            f"非视距AoI={summary['nlos_mean_aoi']:.2f}  "
            f"有遮挡的回合={summary['n_episodes_with_nlos']:.0f}"
        )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"几何信道结果已写入 {csv_path}")


if __name__ == "__main__":
    main()
