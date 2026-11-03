"""Episode-boundary training checkpoints.

Exact continuation requires replay, RNGs, optimizer state and unchanged training
configuration on the same software/device. Cross-device loading is supported,
but numerical/bitwise equivalence is not promised. Legacy model-only checkpoints
remain usable for evaluation, never silently treated as resumable training.
Only load trusted local checkpoints: torch's pickle loader executes Python data.
"""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_random_state(algo, rng: np.random.Generator) -> dict:
    state = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "trainer": copy.deepcopy(rng.bit_generator.state),
        "ou": [{"state": noise.state.copy(), "rng": copy.deepcopy(noise._rng.bit_generator.state)}
               for noise in getattr(algo, "_ou", [])],
    }
    if hasattr(algo, "_rng"):
        state["algorithm"] = copy.deepcopy(algo._rng.bit_generator.state)
    if algo.device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    if algo.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state: dict, algo, rng: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    rng.bit_generator.state = copy.deepcopy(state["trainer"])
    if "algorithm" in state:
        algo._rng.bit_generator.state = copy.deepcopy(state["algorithm"])
    for noise, saved in zip(getattr(algo, "_ou", []), state["ou"]):
        noise.state = saved["state"].copy()
        noise._rng.bit_generator.state = copy.deepcopy(saved["rng"])
    if algo.device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    if algo.device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def validate_resume_config(saved: dict, current: dict) -> None:
    """Reject silently changed dynamics, hyperparameters or random seeds."""
    for key in ("algo", "seed", "env"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"resume changes {key}; use a separate experiment instead")
    operational = {"episodes", "run_dir", "log_dir", "resume_from", "stop_after_episode",
                   "save_every", "log_update_every", "checkpoint_replay"}
    old = {k: v for k, v in saved["train"].items() if k not in operational}
    new = {k: v for k, v in current["train"].items() if k not in operational}
    if old != new:
        raise ValueError("resume changes training hyperparameters")
