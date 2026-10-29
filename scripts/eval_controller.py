"""Roll out a scripted route. No network.

Used for the oracle arm: the leader flies the route and followers track the
true slot. The CSV matches scripts/sweep_geometry.py so the same tables and
figures can include it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.formation_env import UAVFormationEnv, env_config_from_dict
from metrics import summarize_episodes
from utils.config import load_config
from utils.rollout import evaluate_agent
from utils.seeding import set_seed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the scripted route controller.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--p0-values", default="1.0,0.6,0.3")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    env = UAVFormationEnv(env_config_from_dict(cfg["env"]))
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zeros = np.zeros((env.n_agents, env.action_dim), dtype=np.float32)

    def act(_observation):
        return zeros

    rows = []
    for p0 in [float(part) for part in args.p0_values.split(",") if part.strip()]:
        episodes = evaluate_agent(
            env,
            act,
            n_episodes=args.episodes,
            base_seed=args.seed + int(round(p0 * 1000)),
            options={
                "channel_mode": "distance",
                "p0": p0,
                "randomize_channel": False,
                "use_blockage": bool(cfg["env"].get("use_blockage", True)),
            },
        )
        summary = summarize_episodes(episodes)
        summary.update({"p0": p0, "channel_mode": "distance", "algo": "route_oracle"})
        rows.append(summary)
        print(
            f"oracle p0={p0:.2f}  航路={summary['waypoint_fraction']:.2f}  "
            f"近顶高度={summary['near_roof_mean_leader_z']:.1f}  "
            f"高楼高度={summary['tall_mean_leader_z']:.1f}  "
            f"远顶高度={summary['far_roof_mean_leader_z']:.1f}",
            flush=True,
        )
    with (out / "geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
