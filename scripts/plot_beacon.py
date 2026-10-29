"""Neighbour-beacon period and sender rule, against no forwarding and the free upper bound."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["none", "b5", "b2", "b1:random", "b1", "free"]
PRETTY = {
    "none": "No forward",
    "b5": "Beacon / 5",
    "b2": "Beacon / 2",
    "b1:random": "Random / 1",
    "b1": "Freshest / 1",
    "free": "All four / 1",
}
COLORS = ["0.65", "tab:cyan", "tab:blue", "tab:orange", "tab:green", "tab:purple"]


def cvar90(x: np.ndarray) -> float:
    k = max(1, len(x) // 10)
    return float(np.mean(np.sort(x)[-k:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv_path")
    parser.add_argument("--out", default="results/relay_beacon/beacon.png")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.csv_path, encoding="utf-8")))
    cells = sorted(
        {(float(r["jink"]), float(r["nlos_factor"]), float(r["p0"])) for r in rows},
        key=lambda c: (c[0], -c[1], -c[2]),
    )
    data = {}
    for row in rows:
        cell = (float(row["jink"]), float(row["nlos_factor"]), float(row["p0"]))
        data.setdefault(cell, {})[row["trigger"]] = np.array(json.loads(row["e3_episodes"]))
    labels = [f"m{c[0]:g}\nf{c[1]:g}\np{c[2]:g}" for c in cells]
    x = np.arange(len(cells))
    width = 0.8 / len(ORDER)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    for k, arm in enumerate(ORDER):
        means = [data[c][arm].mean() for c in cells]
        ses = [data[c][arm].std(ddof=1) / np.sqrt(len(data[c][arm])) for c in cells]
        tails = [cvar90(data[c][arm]) for c in cells]
        axes[0].bar(x + (k - 2.5) * width, means, width, yerr=ses, color=COLORS[k], label=PRETTY[arm], capsize=2)
        axes[1].bar(x + (k - 2.5) * width, tails, width, color=COLORS[k], label=PRETTY[arm])
    axes[0].set_ylabel("Mean 3D slot error (m)")
    axes[1].set_ylabel("Worst 10% of episodes (m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].grid(axis="y", alpha=0.3)
    n = len(next(iter(data[cells[0]].values())))
    fig.suptitle(
        f"One neighbour beacon, leader still broadcasts every step. Random canyons, {n} paired episodes."
    )
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(args.out)


if __name__ == "__main__":
    main()
