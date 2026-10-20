"""Paper-style figures from the CSVs this repo actually writes.

Labels are in English on purpose: the target venues (CCGrid / WoWMoM) want
English figures, and the default Linux fonts here do not ship CJK glyphs.

Examples
--------
    python scripts/plot_results.py training --csv runs/<run>/metrics.csv --out figs
    python scripts/plot_results.py sweep --csv results/sweep/sweep.csv --out figs
    python scripts/plot_results.py trajectory --npz results/traj.npz --out figs
    python scripts/plot_results.py motivation --csv results/motivation/slot_std_vs_lost_steps.csv --out figs
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows: list[dict], key: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(key, "")
        if raw is None or raw == "":
            values.append(math.nan)
        else:
            values.append(float(raw))
    return np.asarray(values, dtype=np.float64)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    valid = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, valid])


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {out / (stem + '.png')}")


def plot_training(csv_path: Path, out: Path, window: int) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    episode = _column(rows, "episode")
    series = [
        ("return_sum", "Episode return (sum over agents)"),
        ("success", "Success (1 = success)"),
        ("collision", "Collision (1 = collision)"),
        ("formation_keeping_rate", "Formation-keeping rate"),
        ("mean_formation_error", "Mean formation error (m)"),
        ("mean_aoi", "Mean AoI (steps)"),
        ("packet_success_rate", "Packet success rate"),
    ]
    fig, axes = plt.subplots(len(series), 1, figsize=(8, 2.1 * len(series)), sharex=True)
    for ax, (key, title) in zip(axes, series):
        values = _column(rows, key)
        ax.plot(episode, values, color="#9bb0c7", linewidth=1.0, label="per episode")
        ax.plot(episode, _smooth(values, window), color="#1f4e79", linewidth=2.0, label=f"moving mean ({window})")
        ax.set_ylabel(title)
        ax.legend(loc="best", fontsize=9)
    axes[-1].set_xlabel("Training episode")
    fig.suptitle("Training curves", fontsize=14)
    _save(fig, out, "training_curves")


def plot_sweep(csv_path: Path, out: Path) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    loss = _column(rows, "loss_rate") * 100.0
    panels = [
        ("formation_keeping_rate", "Formation-keeping rate"),
        ("success_rate", "Success rate"),
        ("collision_rate", "Collision rate"),
        ("mean_aoi", "Mean AoI (steps)"),
        ("mean_formation_error", "Mean formation error (m)"),
        ("packet_success_rate", "Empirical packet success"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, (key, title) in zip(axes.ravel(), panels):
        mean = _column(rows, key)
        std_key = key + "_std"
        if std_key in rows[0] and rows[0][std_key] != "":
            std = _column(rows, std_key)
            ax.errorbar(loss, mean, yerr=std, marker="o", color="#1f4e79", capsize=3)
        else:
            ax.plot(loss, mean, marker="o", color="#1f4e79")
        ax.set_title(title)
        ax.set_xlabel("Nominal packet loss (%)")
        ax.set_xlim(-5, 75)
    fig.suptitle("Performance under independent packet loss", fontsize=14)
    _save(fig, out, "loss_rate_curves")


def plot_geometry(csv_path: Path, out: Path) -> None:
    """Formation error on the distance x blockage channel, split by blockage."""
    rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")
    p0 = _column(rows, "p0")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, title, ylabel in (
        (
            axes[0],
            "los_mean_formation_error",
            "Line of sight",
            "Formation error (m)",
        ),
        (
            axes[1],
            "nlos_mean_formation_error",
            "Behind a roof",
            "Formation error (m)",
        ),
    ):
        mean = _column(rows, key)
        std_key = key + "_std"
        if std_key in rows[0] and rows[0][std_key] not in ("", "nan"):
            std = _column(rows, std_key)
            ax.errorbar(p0, mean, yerr=std, marker="o", color="#1f4e79", capsize=3)
        else:
            ax.plot(p0, mean, marker="o", color="#1f4e79")
        ax.set_title(title)
        ax.set_xlabel("p0")
        ax.set_ylabel(ylabel)
    fig.suptitle("Distance x blockage channel", fontsize=14)
    _save(fig, out, "geometry_formation_error")
    if "los_mean_alt_error" in rows[0]:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharex=True)
        for ax, key, title in (
            (axes[0], "los_mean_alt_error", "Open line of sight"),
            (axes[1], "nlos_mean_alt_error", "Behind a roof"),
        ):
            ax.plot(p0, _column(rows, key), marker="o", color="#b85c38")
            ax.set_title(title)
            ax.set_xlabel("p0")
            ax.set_ylabel("Altitude error (m)")
        fig.suptitle("Altitude paid, split by the roof", fontsize=14)
        _save(fig, out, "geometry_altitude_error")


def plot_motivation(csv_path: Path, out: Path) -> None:
    """Slot doubt grows on the lever arm. Position doubt does not tell that story."""
    rows = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for regime, label, color, column in (
        ("open", "Slot, open air", "#1f4e79", "slot_std_max_m"),
        ("nlos_q", "Slot, roof inflates Q", "#b85c38", "slot_std_max_m"),
        ("open", "Leader position, open air", "#7f8c8d", "position_std_mean_m"),
    ):
        chosen = [row for row in rows if row["regime"] == regime]
        ax.plot(
            [float(row["lost_steps"]) for row in chosen],
            [float(row[column]) for row in chosen],
            marker="o",
            color=color,
            label=label,
        )
    if rows:
        los_gap = float(rows[0]["failures_between_successes_los"])
        nlos_gap = float(rows[0]["failures_between_successes_nlos"])
        ax.axvline(
            los_gap,
            color="#1f4e79",
            linestyle=":",
            linewidth=1.0,
            label=f"Losses between successes, open ({los_gap:.2f})",
        )
        ax.axvline(
            nlos_gap,
            color="#b85c38",
            linestyle=":",
            linewidth=1.0,
            label=f"Losses between successes, roof ({nlos_gap:.2f})",
        )
    ax.set_xlabel("Lost packets since the last delivery")
    ax.set_ylabel("Standard deviation (m)")
    ax.set_title("The slot lies on the heading lever arm")
    ax.legend(fontsize=8)
    _save(fig, out, "motivation_slot_std")


def plot_trajectory(npz_path: Path, out: Path) -> None:
    data = np.load(npz_path)
    uav = data["uav_xy"]  # (T, N, 2)
    obstacles = data["obstacle_xy"]  # (T, M, 2)
    radii = data["obstacle_r"]
    goal = data["goal"]
    map_size = data["map_size"]
    fig, ax = plt.subplots(figsize=(8, 6.4))
    ax.set_xlim(0, float(map_size[0]))
    ax.set_ylim(0, float(map_size[1]))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    colors = ["#111111", "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    n_agents = uav.shape[1]
    for i in range(n_agents):
        name = "leader" if i == 0 else f"follower {i}"
        ax.plot(uav[:, i, 0], uav[:, i, 1], color=colors[i % len(colors)], linewidth=1.8, label=name)
        ax.scatter(uav[0, i, 0], uav[0, i, 1], color=colors[i % len(colors)], marker="o", s=30)
        ax.scatter(uav[-1, i, 0], uav[-1, i, 1], color=colors[i % len(colors)], marker="x", s=40)
    if obstacles.size:
        # Draw the obstacle at every few frames so motion is visible.
        step = max(1, obstacles.shape[0] // 8)
        for m in range(obstacles.shape[1]):
            for t in range(0, obstacles.shape[0], step):
                circle = plt.Circle(
                    (obstacles[t, m, 0], obstacles[t, m, 1]),
                    float(radii[m]),
                    fill=False,
                    linestyle="--" if t else "-",
                    linewidth=1.2,
                    edgecolor="#666666",
                )
                ax.add_patch(circle)
    if "waypoints" in data.files:
        wps = data["waypoints"]
        ax.plot(wps[:, 0], wps[:, 1], color="#e6b800", linestyle=":", linewidth=1.2, label="route")
        ax.scatter(wps[:, 0], wps[:, 1], marker="*", s=80, color="#e6b800", zorder=5)
    ax.scatter([goal[0]], [goal[1]], marker="*", s=160, color="#e6b800", zorder=6)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("One evaluation trajectory")
    _save(fig, out, "trajectory")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot training, sweep, or trajectory outputs.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("training")
    train.add_argument("--csv", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--window", type=int, default=5)

    sweep = sub.add_parser("sweep")
    sweep.add_argument("--csv", required=True)
    sweep.add_argument("--out", required=True)

    geometry = sub.add_parser("geometry")
    geometry.add_argument("--csv", required=True)
    geometry.add_argument("--out", required=True)

    traj = sub.add_parser("trajectory")
    traj.add_argument("--npz", required=True)
    traj.add_argument("--out", required=True)

    motivation = sub.add_parser("motivation")
    motivation.add_argument("--csv", required=True)
    motivation.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    _style()
    if args.command == "training":
        plot_training(Path(args.csv), Path(args.out), args.window)
    elif args.command == "sweep":
        plot_sweep(Path(args.csv), Path(args.out))
    elif args.command == "geometry":
        plot_geometry(Path(args.csv), Path(args.out))
    elif args.command == "motivation":
        plot_motivation(Path(args.csv), Path(args.out))
    else:
        plot_trajectory(Path(args.npz), Path(args.out))


if __name__ == "__main__":
    main()
