"""Write the motivation table: slot doubt against lost packets.

No policy is trained. The table is the filter and the channel at the
default configuration. Quote it as model behavior, not as a result.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.story import coast_curves, representative_link


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motivation numbers for slot uncertainty.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--out", default="results/motivation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    link = representative_link()
    rows = coast_curves(n_steps=args.steps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "slot_std_vs_lost_steps.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            stored = dict(row)
            stored["regime"] = row["regime"]
            writer.writerow(stored)
    open_6 = next(row for row in rows if row["regime"] == "open" and row["lost_steps"] == 6)
    nlos_6 = next(row for row in rows if row["regime"] == "nlos_q" and row["lost_steps"] == 6)
    print(
        f"40 m 视距 p={link['p_los']:.3f}，两次成功之间平均丢 "
        f"{link['failures_between_successes_los']:.2f} 个包"
    )
    print(
        f"40 m 屋顶挡住 p={link['p_nlos']:.3f}，两次成功之间平均丢 "
        f"{link['failures_between_successes_nlos']:.2f} 个包"
    )
    print(
        f"前排力臂 {open_6['lever_arm_m']:.1f} m。空中丢 6 步：槽位最大轴 "
        f"{open_6['slot_std_max_m']:.2f} m，三轴平均 {open_6['slot_std_mean_m']:.2f} m，"
        f"领队位置 {open_6['position_std_mean_m']:.2f} m，航向均值 {open_6['heading_mean_rad']:.3f}"
    )
    print(
        f"同样 6 步、过程噪声乘 {nlos_6['q_scale']:.0f}：槽位最大轴 "
        f"{nlos_6['slot_std_max_m']:.2f} m，三轴平均 {nlos_6['slot_std_mean_m']:.2f} m"
    )
    print(f"动机表已写入 {csv_path}")


if __name__ == "__main__":
    main()
