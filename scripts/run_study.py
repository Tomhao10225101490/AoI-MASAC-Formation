"""Train and evaluate every enabled arm of experiments/mac_study.yaml.

A finished arm is a checkpoint plus a geometry sweep that uses the same
evaluation seeds for every arm. Re-running the script skips work that is
already on disk.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import deep_update, load_config


def _load_protocol(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def _fmt_meter(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:
        return "—"
    return f"{number:.0f}"


def _bar(done: int, total: int, width: int = 18) -> str:
    total = max(int(total), 1)
    done = max(0, min(int(done), total))
    fill = int(round(width * done / total))
    return "#" * fill + "-" * (width - fill)


def _last_metric(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            last = row
    return last


def _snapshot_table(out: Path) -> str:
    """One line per arm from the latest training episode, so the board shows results."""
    runs = out / "runs"
    if not runs.is_dir():
        return "还没有训练回合。"
    by_arm: dict[str, list[dict[str, str]]] = {}
    for metrics in runs.glob("*/seed_*/metrics.csv"):
        row = _last_metric(metrics)
        if row is None:
            continue
        by_arm.setdefault(metrics.parents[1].name, []).append(row)

    def mean(rows: list[dict[str, str]], key: str, digits: int = 0) -> str:
        vals = []
        for row in rows:
            try:
                number = float(row.get(key, ""))
            except ValueError:
                continue
            if number == number:
                vals.append(number)
        if not vals:
            return "—"
        return f"{sum(vals) / len(vals):.{digits}f}"

    lines = ["臂              种子  编队误差  楼后误差  近顶  高楼  远顶  航路"]
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        lines.append(
            f"{arm:16} {len(rows):2d}   {mean(rows, 'mean_formation_error'):>6}   "
            f"{mean(rows, 'nlos_mean_formation_error'):>6}   {mean(rows, 'near_roof_mean_alt_error'):>4}  "
            f"{mean(rows, 'tall_mean_alt_error'):>4}  {mean(rows, 'far_roof_mean_alt_error'):>4}  "
            f"{mean(rows, 'waypoint_fraction', 2):>4}"
        )
    return "\n".join(lines)


def _print_board(out: Path, jobs: list[dict], active: list, elapsed: float) -> None:
    done = sum((job["eval_dir"] / "geometry.csv").is_file() for job in jobs)
    lines = [
        f"进度 [{_bar(done, len(jobs), 24)}] {done}/{len(jobs)} 组    已用 {int(elapsed) // 60} 分 {int(elapsed) % 60:02d} 秒"
    ]
    if not active:
        lines.append("等待下一组…")
    for job, _proc, _handle, stage, _started in active:
        live = {}
        path = job["run_dir"] / "live.json"
        if path.is_file():
            try:
                live = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                live = {}
        if stage == "train" and live:
            lines.append(
                f"{job['arm']['id']:14} 种子 {job['seed']}  "
                f"回合 [{_bar(live.get('episode', 0), live.get('episodes', 1))}] "
                f"{live.get('episode')}/{live.get('episodes')}  "
                f"步 [{_bar(live.get('step', 0), live.get('max_steps', 1), 12)}] "
                f"{live.get('step')}/{live.get('max_steps')}  "
                f"航路 {live.get('waypoints_done')}/{live.get('n_waypoints')}  "
                f"回报 {_fmt_meter(live.get('return'))}  "
                f"楼后编队 {_fmt_meter(live.get('nlos_form'))}"
            )
            lines.append(
                f"{'':14} 高度误差  近顶 {_fmt_meter(live.get('near_roof_alt_error'))} m   "
                f"高楼 {_fmt_meter(live.get('tall_alt_error'))} m   "
                f"远顶 {_fmt_meter(live.get('far_roof_alt_error'))} m"
            )
        else:
            lines.append(f"{job['arm']['id']:14} 种子 {job['seed']}  阶段 {stage}")
    lines.append("")
    lines.append(_snapshot_table(out))
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    (out / "board.txt").write_text(text + "\n", encoding="utf-8")


def _ensure_bench(out: Path) -> dict:
    bench_path = out / "bench.json"
    shared = ROOT / "results" / "study" / "bench.json"
    if not bench_path.is_file() and shared.is_file():
        bench_path.write_text(shared.read_text(encoding="utf-8"), encoding="utf-8")
    if not bench_path.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_device.py")], cwd=ROOT, check=True)
    return json.loads(bench_path.read_text(encoding="utf-8"))


def _device_plan(protocol: dict, bench: dict) -> tuple[str, int, int]:
    choice = bench["choice"]
    device = str(protocol.get("device", "auto"))
    if device in ("", "auto"):
        device = str(choice["device"])
    workers = int(protocol.get("workers", 0))
    if workers <= 0:
        workers = int(choice["workers"]) if device == choice["device"] else 1
    threads = int(choice["torch_threads"]) if device == "cpu" else max(2, int(choice.get("torch_threads", 4)))
    if device == "mps":
        workers = min(workers, 1)
        threads = 4
    if device == "cpu" and workers > 1:
        threads = 2
    return device, workers, threads


def _write_arm_config(protocol: dict, arm: dict, seed: int, device: str, dest: Path) -> None:
    cfg = load_config(ROOT / arm["config"])
    overlay: dict = {"train": protocol["train"], "seed": seed, "device": device}
    if protocol.get("env"):
        overlay["env"] = protocol["env"]
    cfg = deep_update(cfg, overlay)
    if arm.get("env"):
        cfg = deep_update(cfg, {"env": arm["env"]})
    cfg.pop("_config_path", None)
    cfg.pop("_project_root", None)
    cfg["study_arm"] = arm["id"]
    cfg["study_role"] = arm["role"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _prepare_run_dir(path: Path) -> None:
    if (path / "ckpt_final.pt").is_file():
        return
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path.rename(path.with_name(path.name + f"_incomplete_{stamp}"))


def _popen(cmd: list[str], log_path: Path, env: dict) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return proc, handle


def _child_env(threads: int) -> dict:
    env = os.environ.copy()
    env["AOI_TORCH_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    return env


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    protocol_path = Path(args[0]) if args else ROOT / "experiments" / "mac_study.yaml"
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    (ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
    protocol = _load_protocol(protocol_path)
    out = ROOT / protocol.get("out", "results/study")
    out.mkdir(parents=True, exist_ok=True)
    bench = _ensure_bench(out)
    device, workers, threads = _device_plan(protocol, bench)
    arms = [arm for arm in protocol["arms"] if arm.get("enabled", True)]
    seeds = [int(s) for s in protocol["seeds"]]
    plan = {
        "protocol": str(protocol_path.relative_to(ROOT)),
        "device": device,
        "workers": workers,
        "torch_threads": threads,
        "seeds": seeds,
        "episodes": int(protocol["train"]["episodes"]),
        "arms": [arm["id"] for arm in arms],
        "claim": protocol.get("claim", "").strip(),
        "bench_choice": bench["choice"],
        "machine": bench.get("machine", {}),
    }
    (out / "protocol.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(protocol_path, out / "study.yaml")
    print(
        f"研究 {plan['protocol']}  device={device}  并行={workers}  "
        f"线程={threads}  回合={plan['episodes']}  臂={','.join(plan['arms'])}",
        flush=True,
    )

    jobs = []
    for arm in arms:
        for seed in seeds:
            run_dir = out / "runs" / arm["id"] / f"seed_{seed}"
            eval_dir = out / "eval" / arm["id"] / f"seed_{seed}"
            jobs.append({"arm": arm, "seed": seed, "run_dir": run_dir, "eval_dir": eval_dir})

    env = _child_env(threads)
    pending = list(jobs)
    active: list[tuple[dict, subprocess.Popen, object, str]] = []
    failures = []
    log = out / "jobs.jsonl"
    board_t0 = time.time()

    def record(job: dict, stage: str, code: int, started: float) -> None:
        row = {
            "arm": job["arm"]["id"],
            "seed": job["seed"],
            "stage": stage,
            "returncode": code,
            "seconds": round(time.time() - started, 3),
        }
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if code != 0:
            failures.append(row)
            print(f"失败 {row['arm']} seed={row['seed']} 阶段={stage} code={code}")
        else:
            print(f"完成 {row['arm']} seed={row['seed']} 阶段={stage} 用时={row['seconds']:.0f}s")

    while pending or active:
        while pending and len(active) < workers:
            job = pending.pop(0)
            stage, cmd = _next_command(job, protocol, device, out)
            if cmd is None:
                print(f"跳过 {job['arm']['id']} seed={job['seed']}（评估已在）")
                continue
            print(f"启动 {job['arm']['id']} seed={job['seed']} 阶段={stage}")
            proc, handle = _popen(cmd, out / "logs" / f"{job['arm']['id']}_seed{job['seed']}_{stage}.log", env)
            active.append((job, proc, handle, stage, time.time()))
        still = []
        for job, proc, handle, stage, started in active:
            code = proc.poll()
            if code is None:
                still.append((job, proc, handle, stage, started))
                continue
            handle.close()
            record(job, stage, code, started)
            if code == 0 and stage == "train":
                pending.insert(0, job)
            elif code == 0 and stage == "eval" and job.get("after_eval"):
                job["traj_cmd"] = job.pop("after_eval")
                pending.insert(0, job)
        active = still
        _print_board(out, jobs, active, time.time() - board_t0)
        if active:
            time.sleep(3)

    agg = subprocess.run([sys.executable, str(ROOT / "scripts" / "aggregate_study.py"), str(out)], cwd=ROOT)
    analysis = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "analyze_study.py"), str(out)],
        cwd=ROOT,
    )
    if failures or agg.returncode != 0 or analysis.returncode != 0:
        print(f"研究结束，但有 {len(failures)} 个任务失败。日志在 {out / 'logs'}", flush=True)
        sys.exit(1)
    print(f"研究结束。表、图和分析在 {out}", flush=True)


def _traj_command(job: dict, protocol: dict, device: str) -> list[str]:
    p0 = float(protocol["eval"]["trajectory_p0"])
    return [
        sys.executable,
        str(ROOT / "evaluate.py"),
        "--checkpoint",
        str(job["run_dir"] / "ckpt_final.pt"),
        "--episodes",
        "1",
        "--seed",
        "4200",
        "--device",
        device,
        "--channel-mode",
        "distance",
        "--p0",
        str(p0),
        "--save-traj",
        str(job["eval_dir"] / "traj_p0_0.6.npz"),
        "--out",
        str(job["eval_dir"] / "traj_episode"),
    ]


def _next_command(job: dict, protocol: dict, device: str, out: Path) -> tuple[str | None, list[str] | None]:
    arm = job["arm"]
    seed = job["seed"]
    run_dir = job["run_dir"]
    eval_dir = job["eval_dir"]
    geometry = eval_dir / "geometry.csv"
    traj_path = eval_dir / "traj_p0_0.6.npz"
    want_traj = seed == int(protocol["seeds"][0]) and arm.get("train", True)
    if not arm.get("train", True):
        if geometry.is_file():
            return None, None
        cfg_path = out / "configs" / f"{arm['id']}_seed{seed}.yaml"
        _write_arm_config(protocol, arm, seed, device, cfg_path)
        p0s = ",".join(str(v) for v in protocol["eval"]["p0_values"])
        return "oracle", [
            sys.executable,
            str(ROOT / "scripts" / "eval_controller.py"),
            "--config",
            str(cfg_path),
            "--out",
            str(eval_dir),
            "--episodes",
            str(int(protocol["eval"]["episodes"])),
            "--seed",
            str(int(protocol["eval"]["seed"]) + seed),
            "--p0-values",
            p0s,
        ]
    if job.get("traj_cmd"):
        if traj_path.is_file():
            return None, None
        return "traj", job["traj_cmd"]
    if (run_dir / "ckpt_final.pt").is_file() and geometry.is_file():
        if want_traj and not traj_path.is_file():
            return "traj", _traj_command(job, protocol, device)
        return None, None
    if not (run_dir / "ckpt_final.pt").is_file():
        _prepare_run_dir(run_dir)
        cfg_path = out / "configs" / f"{arm['id']}_seed{seed}.yaml"
        _write_arm_config(protocol, arm, seed, device, cfg_path)
        return "train", [
            sys.executable,
            str(ROOT / "train.py"),
            "--config",
            str(cfg_path),
            "--seed",
            str(seed),
            "--device",
            device,
            "--run-dir",
            str(run_dir),
        ]
    p0s = ",".join(str(v) for v in protocol["eval"]["p0_values"])
    if want_traj:
        job["after_eval"] = _traj_command(job, protocol, device)
    return "eval", [
        sys.executable,
        str(ROOT / "scripts" / "sweep_geometry.py"),
        "--checkpoint",
        str(run_dir / "ckpt_final.pt"),
        "--episodes",
        str(int(protocol["eval"]["episodes"])),
        "--seed",
        str(int(protocol["eval"]["seed"])),
        "--device",
        device,
        "--p0-values",
        p0s,
        "--out",
        str(eval_dir),
    ]


if __name__ == "__main__":
    main()
