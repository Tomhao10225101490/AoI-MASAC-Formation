"""Figures for the roof-split study. Labels are English, for the paper."""

from __future__ import annotations

import csv
import json
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

COLORS = {
    "belief": "#1f4e79",
    "belief_point": "#b85c38",
    "no_blockage": "#2d6a4f",
    "aoi": "#2a9d8f",
    "extrapolate": "#c4a35a",
    "hold": "#6c757d",
    "belief_matd3": "#6a4c93",
    "belief_maddpg": "#bc6c25",
    "route_oracle": "#111111",
}

READING = {
    "altitude_by_roof": "主图。p0=0.6，距离乘遮挡。左列是僚机绝对高度，右列是离开平槽位的高度误差。可飞越的屋顶上，Belief 应该高于 No std；超过升限的楼上，两者都不该往 120 m 的升限爬。No blockage 若也爬，爬的是楼本身，不是在买测量。",
    "geometry_altitude": "同一对比沿 p0 展开。阴影是种子之间的样本标准差。",
    "training_altitude": "训练中途的固定信道探针。横轴是训练回合，不是评估回合。曲线还在动，说明这个预算没有把策略训完。",
    "primary_table": "p0=0.6 的主表，格子是种子间均值 ± 样本标准差。",
    "trajectories": "每种方法的 seed 0，同一条评估随机数，p0=0.6。下排是侧视，红柱是高于升限的楼，灰柱是可以飞越的屋顶，虚线是升限。",
    "motivation_slot_std": "没有策略。槽位标准差沿编队力臂涨，领队位置标准差不讲这件事。楼后过程噪声放大到约 22 m。这张图可以引用，但要写明没有训练。",
    "route_acts": "领队按航路飞过三栋楼时，僚机离开真实槽位的高度误差。近顶和远顶是买得回视线的屋顶，高楼超过升限。Belief 若有用，误差应该小于 No std，而且只出现在两座矮屋顶上。",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
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


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _labels(out: Path) -> dict[str, str]:
    labels = {}
    import yaml

    for name in ("study.yaml", "mac_study.yaml"):
        protocol = out / name
        if not protocol.is_file():
            continue
        data = yaml.safe_load(protocol.read_text(encoding="utf-8"))
        for arm in data.get("arms", []):
            labels[arm["id"]] = arm.get("label", arm["id"])
        break
    return labels


def _f(text: str) -> float:
    if text is None or text == "":
        return float("nan")
    return float(text)


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {out / (stem + '.png')}")


def _arms(rows: list[dict], labels: dict[str, str]) -> list[str]:
    seen = []
    for row in rows:
        if row["arm"] not in seen:
            seen.append(row["arm"])
    order = list(labels)
    seen.sort(key=lambda arm: order.index(arm) if arm in order else 99)
    return seen


def _lookup(rows: list[dict], arm: str, metric: str, p0: float = 0.6) -> tuple[float, float]:
    for row in rows:
        if row["arm"] == arm and row["metric"] == metric and abs(_f(row["p0"]) - p0) < 1e-6:
            return _f(row["mean"]), _f(row["std"])
    return float("nan"), float("nan")


def plot_route_acts(out: Path, labels: dict[str, str]) -> None:
    rows = _read(out / "tables" / "geometry_across_seeds.csv")
    arms = _arms(rows, labels)
    spec = [
        ("near_roof_mean_alt_error", "50 m roof"),
        ("tall_mean_alt_error", "160 m building"),
        ("far_roof_mean_alt_error", "70 m roof"),
    ]
    if not arms or not any(row["metric"] == spec[0][0] for row in rows):
        return
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(spec))
    width = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        means, stds = [], []
        for metric, _name in spec:
            mean, std = _lookup(rows, arm, metric)
            means.append(mean)
            stds.append(0.0 if not math.isfinite(std) else std)
        offset = (i - (len(arms) - 1) / 2) * width
        ax.bar(
            x + offset,
            [v if math.isfinite(v) else np.nan for v in means],
            width=width,
            yerr=stds,
            capsize=2,
            color=COLORS.get(arm, "#333333"),
            label=labels.get(arm, arm),
            error_kw={"elinewidth": 0.8},
        )
    ax.set_xticks(x, [name for _metric, name in spec])
    ax.set_ylabel("Altitude error (m)")
    ax.set_title("While passing each route building, p0 = 0.6")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out / "figs", "route_acts")


def plot_altitude_bars(out: Path, labels: dict[str, str]) -> None:
    rows = _read(out / "tables" / "geometry_across_seeds.csv")
    arms = _arms(rows, labels)
    if not arms:
        return
    groups = [
        ("los_mean_z", "Open air"),
        ("overfly_mean_z", "Overflyable roof"),
        ("impassable_mean_z", "Above ceiling"),
    ]
    prices = [
        ("los_mean_alt_error", "Open air"),
        ("overfly_mean_alt_error", "Overflyable roof"),
        ("impassable_mean_alt_error", "Above ceiling"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=False)
    for ax, spec, ylabel, title in (
        (axes[0], groups, "Follower altitude (m)", "Where the formation flies"),
        (axes[1], prices, "Altitude error (m)", "Price against the flat slot"),
    ):
        x = np.arange(len(spec))
        width = 0.8 / max(len(arms), 1)
        for i, arm in enumerate(arms):
            means, stds = [], []
            for metric, _name in spec:
                mean, std = _lookup(rows, arm, metric)
                means.append(mean)
                stds.append(0.0 if not math.isfinite(std) else std)
            offset = (i - (len(arms) - 1) / 2) * width
            ax.bar(
                x + offset,
                [v if math.isfinite(v) else np.nan for v in means],
                width=width,
                yerr=stds,
                capsize=2,
                color=COLORS.get(arm, "#333333"),
                label=labels.get(arm, arm),
                error_kw={"elinewidth": 0.8},
            )
        ax.set_xticks(x, [name for _metric, name in spec])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].axhline(120, color="#c0392b", linestyle="--", linewidth=0.8, label="Ceiling")
    axes[1].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("Altitude at p0 = 0.6, mean across seeds", fontsize=13)
    _save(fig, out / "figs", "altitude_by_roof")


def plot_geometry_lines(out: Path, labels: dict[str, str]) -> None:
    rows = _read(out / "tables" / "geometry_across_seeds.csv")
    arms = _arms(rows, labels)
    if not arms:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for ax, metric, title in (
        (axes[0], "overfly_mean_z", "Overflyable roof"),
        (axes[1], "impassable_mean_z", "Building above the ceiling"),
    ):
        for arm in arms:
            chosen = [row for row in rows if row["arm"] == arm and row["metric"] == metric]
            chosen.sort(key=lambda row: _f(row["p0"]))
            if not chosen:
                continue
            xs = np.array([_f(row["p0"]) for row in chosen])
            ys = np.array([_f(row["mean"]) for row in chosen])
            ss = np.array([_f(row["std"]) for row in chosen])
            color = COLORS.get(arm, "#333333")
            ax.plot(xs, ys, marker="o", color=color, label=labels.get(arm, arm), linewidth=1.8)
            ax.fill_between(xs, ys - ss, ys + ss, color=color, alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("p0")
        ax.set_ylabel("Follower altitude (m)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("The roof split along the training channel", fontsize=13)
    _save(fig, out / "figs", "geometry_altitude")


def plot_training(out: Path, labels: dict[str, str]) -> None:
    rows = _read(out / "tables" / "eval_across_seeds.csv")
    arms = _arms(rows, labels)
    if not arms:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharex=True)
    for ax, metric, title in (
        (axes[0], "overfly_mean_z", "Probe: overflyable roof"),
        (axes[1], "impassable_mean_z", "Probe: above the ceiling"),
    ):
        for arm in arms:
            chosen = [row for row in rows if row["arm"] == arm and row["metric"] == metric]
            chosen.sort(key=lambda row: int(float(row["episode"])))
            if not chosen:
                continue
            xs = np.array([int(float(row["episode"])) for row in chosen], dtype=float)
            ys = np.array([_f(row["mean"]) for row in chosen])
            ss = np.array([_f(row["std"]) for row in chosen])
            color = COLORS.get(arm, "#333333")
            ax.plot(xs, ys, marker="o", color=color, label=labels.get(arm, arm), linewidth=1.6)
            ax.fill_between(xs, ys - ss, ys + ss, color=color, alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Follower altitude (m)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("In-training probe, mean across seeds", fontsize=13)
    _save(fig, out / "figs", "training_altitude")


def plot_table(out: Path, labels: dict[str, str]) -> None:
    rows = _read(out / "tables" / "primary_p0_0.6.csv")
    if not rows:
        return
    columns = [
        ("success_rate", "Success"),
        ("collision_rate", "Collision"),
        ("waypoint_fraction", "Waypoints"),
        ("overfly_mean_z", "Overfly z"),
        ("impassable_mean_z", "Tall z"),
        ("los_mean_alt_error", "LOS alt err"),
        ("overfly_clear_rate", "Roof clear"),
    ]
    cell = []
    row_labels = []
    for row in rows:
        row_labels.append(labels.get(row["arm"], row["arm"]))
        one = []
        for key, _title in columns:
            mean = _f(row.get(f"{key}_mean", ""))
            std = _f(row.get(f"{key}_std", ""))
            if not math.isfinite(mean):
                one.append("—")
            elif abs(mean) <= 1.5:
                one.append(f"{mean:.2f}±{std:.2f}")
            else:
                one.append(f"{mean:.1f}±{std:.1f}")
        cell.append(one)
    fig, ax = plt.subplots(figsize=(12.2, 0.55 * len(rows) + 1.4))
    ax.axis("off")
    table = ax.table(
        cellText=cell,
        rowLabels=row_labels,
        colLabels=[title for _key, title in columns],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title("p0 = 0.6, mean ± std across seeds", pad=12)
    _save(fig, out / "figs", "primary_table")


def plot_trajectories(out: Path, labels: dict[str, str]) -> None:
    paths = sorted((out / "eval").glob("*/seed_*/traj_p0_0.6.npz"))
    if not paths:
        return
    order = list(labels)
    paths.sort(key=lambda path: order.index(path.parents[1].name) if path.parents[1].name in order else 99)
    n = len(paths)
    fig, axes = plt.subplots(n, 2, figsize=(10.2, 2.35 * n), squeeze=False)
    for row, path in enumerate(paths):
        arm = path.parents[1].name
        data = np.load(path)
        _top(axes[row, 0], data)
        _side(axes[row, 1], data)
        axes[row, 0].set_ylabel(labels.get(arm, arm))
    axes[0, 0].set_title("Plan view")
    axes[0, 1].set_title("Side view")
    fig.suptitle("One paired episode at p0 = 0.6", fontsize=13)
    _save(fig, out / "figs", "trajectories")


def _top(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    uav = data["uav_xy"]
    map_size = data["map_size"]
    ax.set_xlim(0, float(map_size[0]))
    ax.set_ylim(0, float(map_size[1]))
    ax.set_aspect("equal")
    colors = ["#111111", "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
    for i in range(uav.shape[1]):
        ax.plot(uav[:, i, 0], uav[:, i, 1], color=colors[i % len(colors)], linewidth=1.2)
    if "obstacle_xy" in data.files:
        radii = data["obstacle_r"]
        obstacles = data["obstacle_xy"][0]
        for m in range(obstacles.shape[0]):
            ax.add_patch(
                plt.Circle(
                    (obstacles[m, 0], obstacles[m, 1]),
                    float(radii[m]),
                    fill=False,
                    linewidth=0.7,
                    edgecolor="#888888",
                )
            )
    if "waypoints" in data.files:
        wps = data["waypoints"]
        ax.plot(wps[:, 0], wps[:, 1], color="#e6b800", linestyle=":", linewidth=1.0)


def _side(ax: plt.Axes, data: np.lib.npyio.NpzFile) -> None:
    uav = data["uav_xy"]
    z = data["uav_z"]
    zmax = float(data["z_max"]) if "z_max" in data.files else 120.0
    if "obstacle_height" in data.files:
        heights = data["obstacle_height"]
        radii = data["obstacle_r"]
        obstacles = data["obstacle_xy"][0]
        for m in range(len(heights)):
            height = float(heights[m])
            if height <= 0.0:
                continue
            color = "#c0392b" if height > zmax else "#95a5a6"
            x = float(obstacles[m, 0])
            radius = float(radii[m])
            ax.add_patch(plt.Rectangle((x - radius, 0.0), 2.0 * radius, height, color=color, alpha=0.35, linewidth=0))
    ax.axhline(zmax, color="#c0392b", linestyle="--", linewidth=0.8)
    colors = ["#111111", "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"]
    for i in range(uav.shape[1]):
        ax.plot(uav[:, i, 0], z[:, i], color=colors[i % len(colors)], linewidth=1.2)
    ax.set_ylim(0, 180)
    ax.set_xlim(0, float(data["map_size"][0]))
    ax.set_xlabel("x (m)")


def present(out: Path) -> None:
    _style()
    labels = _labels(out)
    plot_altitude_bars(out, labels)
    plot_route_acts(out, labels)
    plot_geometry_lines(out, labels)
    plot_training(out, labels)
    plot_table(out, labels)
    plot_trajectories(out, labels)
    index = {
        "claim": "",
        "figures": [],
    }
    protocol = out / "protocol.json"
    if protocol.is_file():
        index["claim"] = json.loads(protocol.read_text(encoding="utf-8")).get("claim", "")
    figs = out / "figs"
    for stem, note in READING.items():
        png = figs / f"{stem}.png"
        if png.is_file():
            index["figures"].append(
                {"stem": stem, "png": str(png), "pdf": str(figs / f"{stem}.pdf"), "read": note}
            )
    index["tables"] = [
        str(out / "tables" / name)
        for name in (
            "primary_p0_0.6.csv",
            "geometry_across_seeds.csv",
            "geometry_by_seed.csv",
            "eval_across_seeds.csv",
            "eval_by_seed.csv",
        )
        if (out / "tables" / name).is_file()
    ]
    (out / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out / 'index.json'}")
