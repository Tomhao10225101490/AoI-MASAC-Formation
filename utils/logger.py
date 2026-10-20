"""CSV logger plus an optional TensorBoard writer.

CSV is the source of truth for the plot scripts. TensorBoard is best-effort:
if the import fails, training still runs and a one-line warning is printed.
"""

from __future__ import annotations

import csv
from pathlib import Path


class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: dict) -> None:
        encoded = {}
        for key in self.fieldnames:
            value = row.get(key, "")
            if isinstance(value, float):
                encoded[key] = f"{value:.6g}"
            else:
                encoded[key] = value
        self._writer.writerow(encoded)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class TrainLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = CsvLogger(
            self.log_dir / "metrics.csv",
            [
                "episode",
                "seed",
                "algo",
                "steps",
                "total_steps",
                "wall_time_sec",
                "update_time_sec",
                "n_updates",
                "return_sum",
                "return_mean",
                "success",
                "collision",
                "obstacle_collision",
                "inter_uav_collision",
                "timeout",
                "formation_keeping_rate",
                "mean_formation_error",
                "mean_aoi",
                "max_aoi",
                "packet_success_rate",
                "mean_outage_length",
                "waypoint_fraction",
                "nlos_fraction",
                "min_clearance",
                "mean_belief_std",
                "p0",
                "critic_loss",
                "actor_loss",
                "alpha",
                "q",
            ],
        )
        self.updates = CsvLogger(
            self.log_dir / "updates.csv",
            ["total_steps", "critic_loss", "actor_loss", "alpha", "q"],
        )
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.log_dir / "tb"))
        except Exception as exc:  # pragma: no cover - depends on the install
            print(f"[warn] TensorBoard disabled ({exc}). CSV logs are still written.")

    def log_episode(self, row: dict) -> None:
        self.episodes.log(row)
        if self._tb is not None:
            step = int(row["total_steps"])
            for key in (
                "return_sum",
                "success",
                "collision",
                "formation_keeping_rate",
                "mean_formation_error",
                "mean_aoi",
                "packet_success_rate",
                "wall_time_sec",
                "critic_loss",
                "actor_loss",
                "alpha",
            ):
                value = row.get(key, "")
                if value == "" or value is None:
                    continue
                self._tb.add_scalar(f"train/{key}", float(value), step)

    def log_update(self, total_steps: int, stats: dict) -> None:
        if not stats:
            return
        row = {"total_steps": total_steps, **stats}
        self.updates.log(row)
        if self._tb is not None:
            for key, value in stats.items():
                self._tb.add_scalar(f"update/{key}", float(value), total_steps)

    def close(self) -> None:
        self.episodes.close()
        self.updates.close()
        if self._tb is not None:
            self._tb.close()
