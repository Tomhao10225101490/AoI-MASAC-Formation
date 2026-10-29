"""Pre-registered comparisons for a finished study directory.

Writes analysis.md next to the tables. A missing arm or an empty split is
reported as missing, not as a win.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.present_study import _f, _read


def _row(rows: list[dict], arm: str, metric: str, p0: float) -> dict | None:
    for row in rows:
        if row["arm"] == arm and row["metric"] == metric and abs(_f(row["p0"]) - p0) < 1e-6:
            return row
    return None


def _fmt(mean: float, std: float, n: int) -> str:
    if n <= 0 or not math.isfinite(mean):
        return "缺样本"
    return f"{mean:.2f} ± {std:.2f} (n={n})"


def analyze(out: Path) -> str:
    protocol_path = out / "study.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    rows = _read(out / "tables" / "geometry_across_seeds.csv")
    lines = [
        "# 预先登记的比较",
        "",
        "数字是评估回合先在种子内平均，再对种子取均值和样本标准差。",
        "n=3 时区间重叠只说明这三粒种子没有分开，不把它写成显著性检验。",
        "",
    ]
    hypotheses = protocol.get("hypotheses") or []
    wins = 0
    tested = 0
    for item in hypotheses:
        tested += 1
        better = _row(rows, item["better"], item["metric"], float(item["p0"]))
        worse = _row(rows, item["worse"], item["metric"], float(item["p0"]))
        lines.append(f"## {item['id']}")
        lines.append("")
        lines.append(item["claim"])
        lines.append("")
        if better is None or worse is None:
            lines.append("结果：缺一边的样本，这条先不下结论。")
            lines.append("")
            continue
        b_mean, b_std, b_n = _f(better["mean"]), _f(better["std"]), int(float(better["n_seeds"]))
        w_mean, w_std, w_n = _f(worse["mean"]), _f(worse["std"]), int(float(worse["n_seeds"]))
        lines.append(f"- {item['better']}: {_fmt(b_mean, b_std, b_n)}")
        lines.append(f"- {item['worse']}: {_fmt(w_mean, w_std, w_n)}")
        if "max_gap" in item:
            gap = b_mean - w_mean
            ok = math.isfinite(gap) and abs(gap) <= float(item["max_gap"])
            lines.append(f"- 高度差 {gap:.2f} m，允许 ±{item['max_gap']} m：{'没有越出' if ok else '越出了'}")
            wins += int(ok)
        else:
            # Lower is better for every error metric in this protocol.
            separated = math.isfinite(b_mean) and math.isfinite(w_mean) and (b_mean + b_std) < (w_mean - w_std)
            direction = b_mean < w_mean
            if separated:
                lines.append("- 结果：均值差距大于两边的种子标准差，方向是更低的误差。")
                wins += 1
            elif direction:
                lines.append("- 结果：均值更低，但种子区间重叠。方向对，还不能单独写成已经分开。")
            else:
                lines.append("- 结果：均值没有更低。这条不支持。")
        lines.append("")
    lines.append("## 总览")
    lines.append("")
    lines.append(f"预先登记的比较里，{tested} 条中有 {wins} 条达到了写在协议里的通过标准。")
    lines.append("通过标准是：误差更低且两个种子区间不重叠；高楼高度只要求两边相差不超过登记的上限。")
    lines.append("几何参照（Oracle）不是学习出来的，用来看跟踪器本身还剩多少误差。")
    text = "\n".join(lines) + "\n"
    (out / "analysis.md").write_text(text, encoding="utf-8")
    print(text)
    return text


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    out = Path(args[0]) if args else ROOT / "results" / "day"
    if not out.is_absolute():
        out = ROOT / out
    if not (out / "tables" / "geometry_across_seeds.csv").is_file():
        print(f"还没有汇总表：{out}")
        sys.exit(1)
    analyze(out)


if __name__ == "__main__":
    main()
