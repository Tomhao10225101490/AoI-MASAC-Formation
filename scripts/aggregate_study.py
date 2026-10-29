"""Collapse per-seed study outputs into tables and figures.

    python scripts/aggregate_study.py results/study
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.present_study import present

ID_COLUMNS = {
    "p0",
    "channel_mode",
    "use_blockage",
    "obs_mode",
    "use_aoi",
    "algo",
    "arm",
    "seed",
    "episode",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(text: str) -> float:
    if text is None or text == "":
        return float("nan")
    return float(text)


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan"), float("nan"), 0
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return mean, math.sqrt(var), len(finite)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key, value in row.items():
                if isinstance(value, float):
                    encoded[key] = "" if math.isnan(value) else f"{value:.6g}"
                else:
                    encoded[key] = value
            writer.writerow(encoded)


def _geometry_rows(out: Path) -> list[dict]:
    rows = []
    for path in sorted((out / "eval").glob("*/seed_*/geometry.csv")):
        arm = path.parents[1].name
        seed = int(path.parent.name.split("_")[1])
        for raw in _read_csv(path):
            item = {"arm": arm, "seed": seed}
            item.update(raw)
            rows.append(item)
    return rows


def _across(by_seed: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in by_seed:
        grouped[(row["arm"], row.get("p0", ""))].append(row)
    numeric = []
    if by_seed:
        for key, value in by_seed[0].items():
            if key in ID_COLUMNS:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            numeric.append(key)
    out_rows = []
    for (arm, p0), group in sorted(grouped.items()):
        for key in numeric:
            mean, std, n = _mean_std([_float(row.get(key, "")) for row in group])
            out_rows.append(
                {
                    "arm": arm,
                    "p0": _float(p0),
                    "metric": key,
                    "n_seeds": n,
                    "mean": mean,
                    "std": std,
                }
            )
    return out_rows


def _eval_rows(out: Path) -> list[dict]:
    rows = []
    for path in sorted((out / "runs").glob("*/seed_*/eval_metrics.csv")):
        arm = path.parents[1].name
        seed = int(path.parent.name.split("_")[1])
        if not path.is_file():
            continue
        for raw in _read_csv(path):
            item = {"arm": arm, "seed": seed}
            item.update(raw)
            rows.append(item)
    return rows


def _eval_across(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    metrics = [key for key in rows[0] if key not in ID_COLUMNS and key != "episode"]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["episode"])].append(row)
    out_rows = []
    for (arm, episode), group in sorted(grouped.items(), key=lambda item: (item[0][0], int(float(item[0][1])))):
        for key in metrics:
            mean, std, n = _mean_std([_float(row.get(key, "")) for row in group])
            out_rows.append(
                {
                    "arm": arm,
                    "episode": int(float(episode)),
                    "metric": key,
                    "n_seeds": n,
                    "mean": mean,
                    "std": std,
                }
            )
    return out_rows


def _primary(across: list[dict], p0: float) -> list[dict]:
    metrics = [
        "success_rate",
        "collision_rate",
        "waypoint_fraction",
        "mean_formation_error",
        "los_mean_alt_error",
        "overfly_mean_alt_error",
        "impassable_mean_alt_error",
        "los_mean_z",
        "overfly_mean_z",
        "impassable_mean_z",
        "overfly_mean_leader_z",
        "impassable_mean_leader_z",
        "overfly_clear_rate",
        "nlos_mean_formation_error",
        "nlos_mean_aoi",
    ]
    arms = []
    for row in across:
        if row["arm"] not in arms and abs(float(row["p0"]) - p0) < 1e-6:
            arms.append(row["arm"])
    table = []
    for arm in arms:
        item = {"arm": arm, "p0": p0}
        for metric in metrics:
            match = [
                row
                for row in across
                if row["arm"] == arm and row["metric"] == metric and abs(float(row["p0"]) - p0) < 1e-6
            ]
            if not match:
                item[f"{metric}_mean"] = float("nan")
                item[f"{metric}_std"] = float("nan")
                item[f"{metric}_n"] = 0
            else:
                item[f"{metric}_mean"] = match[0]["mean"]
                item[f"{metric}_std"] = match[0]["std"]
                item[f"{metric}_n"] = match[0]["n_seeds"]
        table.append(item)
    return table


def aggregate(out: Path) -> None:
    tables = out / "tables"
    by_seed = _geometry_rows(out)
    across = _across(by_seed)
    _write_csv(tables / "geometry_by_seed.csv", by_seed)
    _write_csv(tables / "geometry_across_seeds.csv", across)
    primary = _primary(across, 0.6)
    _write_csv(tables / "primary_p0_0.6.csv", primary)
    eval_rows = _eval_rows(out)
    _write_csv(tables / "eval_by_seed.csv", eval_rows)
    _write_csv(tables / "eval_across_seeds.csv", _eval_across(eval_rows))
    present(out)
    print(f"tables in {tables}")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(args[0]) if args else ROOT / "results" / "study"
    if not out.is_absolute():
        out = ROOT / out
    aggregate(out)


if __name__ == "__main__":
    main()
