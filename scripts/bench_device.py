"""Time one short Belief-MASAC run on CPU and on MPS.

Writes results/study/bench.json. The study runner uses it to pick the
device and the number of parallel jobs. Two CPU jobs can beat one MPS job
when the network is small, so the choice is throughput, not the device name.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import run_training
from utils.config import load_config


def machine() -> dict:
    memory_gb = (
        __import__("os").sysconf("SC_PAGE_SIZE") * __import__("os").sysconf("SC_PHYS_PAGES") / 1024**3
    )
    chip = ""
    cores = ""
    try:
        text = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
            text=True,
            timeout=40,
        )
        for line in text.splitlines():
            if "Chip:" in line and not chip:
                chip = line.split(":", 1)[1].strip()
            if "Total Number of Cores:" in line and not cores:
                cores = line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
        "cores": cores,
        "cpu_count": __import__("os").cpu_count(),
        "memory_gb": round(memory_gb, 2),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def _one(device: str, out_root: Path) -> dict | None:
    if device == "mps" and not torch.backends.mps.is_available():
        return None
    cfg = load_config(ROOT / "configs" / "paper_belief_masac.yaml")
    cfg["device"] = device
    cfg["seed"] = 0
    # Long enough that the batch-256 update, not just the forward, is in the average.
    run_dir = out_root / "bench_runs" / device
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cfg["env"]["max_steps"] = 80
    cfg["env"]["randomize_channel"] = False
    cfg["env"]["p0"] = 0.8
    cfg["train"].update(
        {
            "episodes": 6,
            "warmup_steps": 256,
            "start_steps": 40,
            "batch_size": 256,
            "hidden": 256,
            "buffer_size": 4000,
            "eval_every": 0,
            "save_every": 0,
            "log_update_every": 10_000,
            "log_dir": str(out_root / "bench_runs"),
            "run_dir": str(run_dir),
        }
    )
    print(f"计时 device={device}")
    run_dir = run_training(cfg)
    speed = json.loads((run_dir / "speed.json").read_text(encoding="utf-8"))
    speed["ok"] = True
    return speed


def choose(cpu: dict | None, mps: dict | None, memory_gb: float) -> dict:
    """Pick the queue that finishes more environment steps per wall second."""
    cpu_step = None if cpu is None else cpu.get("mean_step_sec")
    mps_step = None if mps is None else mps.get("mean_step_sec")
    # Two processes on 4 performance cores do not scale by 2.
    cpu_workers = 2 if memory_gb <= 24 else 3
    cpu_throughput = (1.6 / cpu_step) if cpu_step else 0.0
    mps_throughput = (1.0 / mps_step) if mps_step else 0.0
    if mps_throughput > cpu_throughput and mps_step:
        return {
            "device": "mps",
            "workers": 1,
            "torch_threads": 4,
            "reason": "one MPS run steps faster than two CPU runs sharing the performance cores",
            "cpu_throughput_steps_per_s": cpu_throughput,
            "mps_throughput_steps_per_s": mps_throughput,
        }
    return {
        "device": "cpu",
        "workers": cpu_workers,
        "torch_threads": 2,
        "reason": "two CPU runs finish more steps per second than one MPS run",
        "cpu_throughput_steps_per_s": cpu_throughput,
        "mps_throughput_steps_per_s": mps_throughput,
    }


def main() -> None:
    out = ROOT / "results" / "study"
    out.mkdir(parents=True, exist_ok=True)
    info = machine()

    def timed(device: str) -> dict | None:
        try:
            return _one(device, out)
        except Exception as exc:
            print(f"计时失败 device={device}: {exc}")
            return {"ok": False, "error": str(exc), "mean_step_sec": None}

    cpu = timed("cpu")
    mps = timed("mps")
    choice = choose(cpu, mps, float(info["memory_gb"]))
    payload = {"machine": info, "cpu": cpu, "mps": mps, "choice": choice}
    path = out / "bench.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(choice, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
