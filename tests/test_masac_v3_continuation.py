"""Resuming must reproduce sampling, Bellman updates and physical transitions."""
import json
import random

import numpy as np
import pytest
import torch

from scripts.continue_masac_v3 import capture, check_parent, restore_completed_run, save_replay
from uav_planning.shared_masac import SharedMASAC
from uav_planning.structured import StructuredMission


def test_resume_reproduces_next_action_transition_and_update(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(89)
    np.random.seed(89)
    random.seed(89)
    model = SharedMASAC(capacity=96, seed=89)
    env = StructuredMission()
    obs = env.reset(912345)
    rng = np.random.default_rng(89)
    for _ in range(48):
        action = rng.uniform(-1, 1, (3, 3)).astype(np.float32)
        nxt, reward, done, trunc, _ = env.step(action)
        model.push(obs, action, reward, nxt, done)
        obs = nxt
        if done or trunc:
            obs = env.reset(912345)
    model.update(16)
    cfg = dict(steps=48)
    checkpoint = capture(model, env, rng, step=48, episode=0,
                         episode_return=1.23, seed=89, cfg=cfg)
    torch.save(checkpoint, tmp_path / 'checkpoint_48.pt')
    save_replay(model, tmp_path)

    expected_action = model.act(obs)
    expected_next, expected_reward, expected_done, _, _ = env.step(expected_action)
    model.push(obs, expected_action, expected_reward, expected_next, expected_done)
    expected_loss = model.update(16)
    expected_actor = {k: v.clone() for k, v in model.actor.state_dict().items()}
    expected_q = {k: v.clone() for k, v in model.q.state_dict().items()}
    expected_rng = (random.random(), np.random.rand(), rng.random())

    restored = SharedMASAC(capacity=96, seed=1)
    _, restored_env, restored_obs, restored_rng = restore_completed_run(restored, tmp_path, cfg)
    assert np.array_equal(obs, restored_obs)
    action = restored.act(restored_obs)
    assert np.array_equal(expected_action, action)
    nxt, reward, done, _, _ = restored_env.step(action)
    assert np.array_equal(expected_next, nxt)
    assert reward == expected_reward and done == expected_done
    restored.push(restored_obs, action, reward, nxt, done)
    assert expected_loss == restored.update(16)
    for key, value in restored.actor.state_dict().items():
        assert torch.equal(value, expected_actor[key])
    for key, value in restored.q.state_dict().items():
        assert torch.equal(value, expected_q[key])
    assert expected_rng == (random.random(), np.random.rand(), restored_rng.random())


def test_incomplete_parent_cannot_start_duplicate_or_reset(tmp_path):
    with pytest.raises(ValueError, match='complete and error-free'):
        check_parent(tmp_path, [411], 100000)
    (tmp_path / 'completion.json').write_text(json.dumps(dict(complete=True)))
    (tmp_path / 'error.json').write_text(json.dumps(dict(error='interrupted')))
    with pytest.raises(ValueError, match='complete and error-free'):
        check_parent(tmp_path, [411], 100000)
