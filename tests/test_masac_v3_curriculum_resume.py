import json
import random
from pathlib import Path

import numpy as np
import torch

from scripts.continue_masac_v3 import capture, save_replay
from scripts.train_masac_v3_curriculum import new_episode_stats, restore_curriculum
from uav_planning.masac_v3_curriculum import make_episode
from uav_planning.shared_masac import SharedMASAC


def test_curriculum_final_checkpoint_replays_next_update(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/masac_v3/protocol.json').read_text())
    cfg.update(json.loads((root/'configs/masac_v3/curriculum_protocol.json').read_text()))
    cfg.update(smoke=True, steps=160, warmup=0, stages=[{'name':'short_edge','start_step':0}, {'name':'full','start_step':70}])
    (tmp_path/'protocol.json').write_text(json.dumps(cfg))
    directory = tmp_path/'standard_411'
    directory.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(411); np.random.seed(411); random.seed(411)
    exploration = np.random.default_rng(411)
    model = SharedMASAC(capacity=3000, seed=411)
    episode = episode_start = 0
    env, obs, stage = make_episode(411, episode, 0, cfg)
    for step in range(1, 161):
        action = exploration.uniform(-1, 1, (3, 3)).astype(np.float32)
        no, reward, done, trunc, _ = env.step(action)
        model.push(obs, action, reward, no, done)
        obs = no
        if done or trunc:
            episode += 1; episode_start = step
            env, obs, stage = make_episode(411, episode, step, cfg)
    stats = new_episode_stats(env)
    cp = capture(model, env, exploration, step=160, episode=episode,
                 episode_return=1.25, seed=411, cfg=cfg)
    cp.update(curriculum_stage=stage, episode_start_step=episode_start,
              episode_stats=stats, exact_resume=True)
    torch.save(cp, directory/'checkpoint_160.pt')
    save_replay(model, directory)

    # Continue directly, then restore the saved RNGs, episode and replay and do
    # exactly the same next stochastic action, transition and SAC update.
    action = model.act(obs)
    no, reward, done, trunc, info = env.step(action)
    model.push(obs, action, reward, no, done)
    losses = model.update(128)
    draws = (random.random(), np.random.random(), exploration.random())
    restored, restored_env, restored_obs, restored_rng, saved = restore_curriculum(directory)
    assert saved['episode'] == episode and saved['curriculum_stage'] == stage
    assert np.array_equal(saved['episode_stats']['closest_goal_m'], stats['closest_goal_m'])
    assert np.array_equal(restored_obs, obs)
    restored_action = restored.act(restored_obs)
    rno, rreward, rdone, rtrunc, rinfo = restored_env.step(restored_action)
    restored.push(restored_obs, restored_action, rreward, rno, rdone)
    rlosses = restored.update(128)
    rdraws = (random.random(), np.random.random(), restored_rng.random())
    assert np.array_equal(restored_action, action)
    assert np.array_equal(rno, no)
    assert (rreward, rdone, rtrunc, rinfo) == (reward, done, trunc, info)
    assert rlosses == losses and rdraws == draws
    for name in ('actor', 'q', 'target'):
        for key, value in getattr(model, name).state_dict().items():
            assert torch.equal(value, getattr(restored, name).state_dict()[key])
