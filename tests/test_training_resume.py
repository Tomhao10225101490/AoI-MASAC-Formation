"""Resume equivalence verifies learning state, not just loaded inference weights."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from algos.buffer import ReplayBuffer
from train import run_training
from utils.config import load_config


def config(path: Path, algo: str = "belief_masac") -> dict:
    cfg = load_config("configs/smoke.yaml")
    cfg.update(algo=algo, seed=7, device="cpu")
    cfg["env"].update(max_steps=8, n_followers=1, obs_mode="belief")
    cfg["train"].update(episodes=4, run_dir=str(path), hidden=8, batch_size=4,
                        buffer_size=16, start_steps=3, warmup_steps=4,
                        curriculum_perfect_frac=.5, ou_episodes=4,
                        eval_every=2, eval_episodes=1, save_every=1)
    return cfg


def equal_nested(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_nested(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_nested(x, y)
    else:
        assert a == b


def load(path):
    return torch.load(path / "ckpt_final.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("algo", ["belief_masac", "maddpg", "matd3"])
def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch, algo):
    monkeypatch.setenv("AOI_TORCH_THREADS", "1")
    cfg = config(tmp_path / "full", algo)
    full = load(run_training(cfg))
    split = copy.deepcopy(cfg)
    split["train"].update(run_dir=str(tmp_path / "partial"), stop_after_episode=1)
    partial = run_training(split)
    continuation = copy.deepcopy(cfg)
    continuation["train"].update(run_dir=str(tmp_path / "continued"),
                                 resume_from=str(partial / "ckpt_final.pt"))
    resumed = load(run_training(continuation))
    assert resumed["episode"] == full["episode"] == 4
    assert resumed["total_steps"] == full["total_steps"]
    assert resumed["training_state"]["total_updates"] > 0
    equal_nested(resumed["algo"], full["algo"])
    equal_nested(resumed["replay"], full["replay"])
    equal_nested(resumed["training_state"]["random"], full["training_state"]["random"])
    assert resumed["training_state"]["perfect_episodes"] == 2
    # First run's logs survive; continuation starts at the next episode.
    assert len((partial / "metrics.csv").read_text().splitlines()) == 2
    rows = (tmp_path / "continued" / "metrics.csv").read_text().splitlines()
    assert len(rows) == 4
    assert rows[1].startswith("1,")


def test_resume_rejects_model_only_checkpoint(tmp_path):
    cfg = config(tmp_path / "model")
    cfg["train"].update(checkpoint_replay=False, stop_after_episode=1)
    model = run_training(cfg)
    cfg["train"].update(run_dir=str(tmp_path / "resume"), resume_from=str(model / "ckpt_final.pt"))
    with pytest.raises(ValueError, match="evaluation-only"):
        run_training(cfg)


def test_resume_rejects_changed_dynamics(tmp_path):
    cfg = config(tmp_path / "model")
    cfg["train"]["stop_after_episode"] = 1
    model = run_training(cfg)
    cfg["env"]["p0"] = .1
    cfg["train"].update(run_dir=str(tmp_path / "resume"), resume_from=str(model / "ckpt_final.pt"))
    with pytest.raises(ValueError, match="changes env"):
        run_training(cfg)


def test_replay_restore_rejects_incompatible_shape():
    source = ReplayBuffer(8, [3], 4, 2, 1)
    dest = ReplayBuffer(8, [5], 4, 2, 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        dest.load_state_dict(source.state_dict())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_checkpoint_continuation_is_finite(tmp_path, monkeypatch):
    monkeypatch.setenv("AOI_TORCH_THREADS", "1")
    cfg = config(tmp_path / "mps_partial")
    cfg["device"] = "mps"
    cfg["train"].update(episodes=2, stop_after_episode=1, eval_every=0, ou_episodes=0)
    partial = run_training(cfg)
    saved = load(partial)
    assert "mps" in saved["training_state"]["random"]
    cfg["train"].update(run_dir=str(tmp_path / "mps_continued"), stop_after_episode=2,
                        resume_from=str(partial / "ckpt_final.pt"))
    result = load(run_training(cfg))
    assert result["episode"] == 2
    assert result["total_steps"] > saved["total_steps"]
    for actor in result["algo"]["actors"]:
        assert all(torch.isfinite(value).all() for value in actor.values())
