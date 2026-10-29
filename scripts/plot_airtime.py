"""Compare leader-only, forwarding, and the AoI relay under three airtime budgets.

Each CSV is one budget (free, or TDMA with a given period) on the same seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ("none", "mesh+none", "relay:coveraoi:10")
COLORS = {"none": "0.55", "mesh+none": "tab:blue", "relay:coveraoi:10": "tab:green"}
NAMES = {"none": "Leader only", "mesh+none": "Forward, no climb", "relay:coveraoi:10": "AoI relay"}


def cvar90(x: np.ndarray) -> float:
    k = max(1, len(x) // 10)
    return float(np.mean(np.sort(x)[-k:]))


def load(path: str) -> dict:
    data = {}
    for row in csv.DictReader(open(path, encoding="utf-8")):
        cell = (float(row["jink"]), float(row["nlos_factor"]), float(row["p0"]))
        data.setdefault(cell, {})[row["trigger"]] = np.array(json.loads(row["e3_episodes"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/relay_tdma/airtime.png")
    parser.add_argument("labeled", nargs="+", help="label=path.csv")
    args = parser.parse_args()
    budgets = [(part.split("=", 1)[0], load(part.split("=", 1)[1])) for part in args.labeled]
    cells = sorted(
        set.intersection(*(set(b.keys()) for _, b in budgets)),
        key=lambda c: (c[0], -c[1], -c[2]),
    )
    labels = [f"m{c[0]:g}\nf{c[1]:g}\np{c[2]:g}" for c in cells]
    fig, axes = plt.subplots(2, len(budgets), figsize=(5.2 * len(budgets), 7.2), sharey="row")
    if len(budgets) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    x = np.arange(len(cells))
    width = 0.8 / len(ARMS)
    for col, (name, data) in enumerate(budgets):
        for k, arm in enumerate(ARMS):
            means = [data[c][arm].mean() for c in cells]
            ses = [data[c][arm].std(ddof=1) / np.sqrt(len(data[c][arm])) for c in cells]
            tails = [cvar90(data[c][arm]) for c in cells]
            axes[0, col].bar(x + (k - 1) * width, means, width, yerr=ses, color=COLORS[arm], label=NAMES[arm], capsize=2)
            axes[1, col].bar(x + (k - 1) * width, tails, width, color=COLORS[arm], label=NAMES[arm])
        axes[0, col].set_title(name)
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("Mean 3D slot error (m)")
    axes[1, 0].set_ylabel("Worst 10% episodes, CVaR90 (m)")
    axes[0, 0].legend(fontsize=8)
    n = len(next(iter(budgets[0][1][cells[0]].values())))
    fig.suptitle(f"Random street canyons, {n} paired episodes. m manoeuvre, f roof penetration, p base delivery")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(args.out)


if __name__ == "__main__":
    main()
