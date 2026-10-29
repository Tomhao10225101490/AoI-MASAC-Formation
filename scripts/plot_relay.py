"""Figures for the relay study: mean error, tail error, and episode CDF per regime.

Reads one or more voi_probe CSVs that share seeds and cells, so the arms are
paired episode by episode.
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

ARMS = [
    ("none", "No forwarding", "0.55"),
    ("mesh+none", "Forwarding only", "tab:blue"),
    ("relay:fixed", "Fixed relay", "tab:orange"),
    ("relay:coveraoi:10", "AoI-gated cover relay", "tab:green"),
    ("oracle", "True slot", "tab:purple"),
]


def cvar90(x: np.ndarray) -> float:
    k = max(1, len(x) // 10)
    return float(np.mean(np.sort(x)[-k:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csvs", nargs="+")
    parser.add_argument("--out", default="results/relay_tail/relay_story.png")
    parser.add_argument("--cdf-cell", default="0.6,0.01,0.3", help="jink,nlos_factor,p0")
    args = parser.parse_args()

    data: dict[tuple, dict[str, np.ndarray]] = {}
    for path in args.csvs:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            cell = (float(r["jink"]), float(r["nlos_factor"]), float(r["p0"]))
            data.setdefault(cell, {})[r["trigger"]] = np.array(json.loads(r["e3_episodes"]))
    cells = sorted(
        (c for c in data if all(a in data[c] for a, _, _ in ARMS)),
        key=lambda c: (c[0], -c[1], -c[2]),
    )
    labels = [f"m{c[0]:g}\nf{c[1]:g}\np{c[2]:g}" for c in cells]
    x = np.arange(len(cells))
    width = 0.8 / len(ARMS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [2, 2, 1.2]})
    for k, (arm, name, color) in enumerate(ARMS):
        means = [data[c][arm].mean() for c in cells]
        ses = [data[c][arm].std(ddof=1) / np.sqrt(len(data[c][arm])) for c in cells]
        tails = [cvar90(data[c][arm]) for c in cells]
        axes[0].bar(x + (k - 2) * width, means, width, yerr=ses, color=color, label=name, capsize=2)
        axes[1].bar(x + (k - 2) * width, tails, width, color=color, label=name)
    for ax, title in zip(axes[:2], ("Mean 3D slot error (m), ±1 s.e.", "Worst-10% episodes, CVaR90 (m)")):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("m: leader manoeuvre, f: roof penetration factor, p: base delivery p0", fontsize=8)

    cdf_cell = tuple(float(v) for v in args.cdf_cell.split(","))
    for arm, name, color in ARMS:
        vals = np.sort(data[cdf_cell][arm])
        axes[2].step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post", color=color, label=name)
    axes[2].set_xlabel("episode mean 3D slot error (m)")
    axes[2].set_ylabel("fraction of episodes")
    axes[2].set_title(f"Episode CDF, m={cdf_cell[0]:g} f={cdf_cell[1]:g} p0={cdf_cell[2]:g}")
    axes[2].grid(alpha=0.3)
    n = len(next(iter(data[cells[0]].values())))
    fig.suptitle(f"Street canyon, scripted policies, {n} paired episodes per bar")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(args.out)


if __name__ == "__main__":
    main()
