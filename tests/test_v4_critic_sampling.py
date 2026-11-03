import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.continue_masac_v3 import capture, restore_completed_run, save_replay
from uav_planning.critic_sampling_diagnostic import CriticSamplingUpdater, ReplayStrata, sample_critic
from uav_planning.shared_masac import SharedMASAC
from uav_planning.structured import StructuredMission


def replay_with_done(labels):
    m = SharedMASAC(capacity=len(labels), seed=3)
    obs = np.zeros((3, 162), np.float32)
    action = np.zeros((3, 3), np.float32)
    for i, label in enumerate(labels):
        m.push(obs+i, action, float(i), obs+i+1, label)
    return m


def test_fixed_strata_weighted_mean_is_unbiased_without_replacement():
    labels = np.array([0]*7 + [1]*3)
    values = np.array([-.3, .2, .9, 1.4, 2.1, 3.3, 4.7, -12., -4., 8.])
    # Exhaust all possible equally likely samples: 3/7 nonterminal and 1/3 terminal.
    estimates = []
    for nonterminal in itertools.combinations(range(7), 3):
        for terminal in range(7, 10):
            ids = list(nonterminal)+[terminal]
            weights = np.array([(.7/.75)]*3 + [(.3/.25)])
            estimates.append(np.mean(weights*values[ids]))
    assert np.mean(estimates) == pytest.approx(values.mean(), abs=1e-12)
    m = replay_with_done(labels)
    strata = ReplayStrata(m)
    ids, weights, info = sample_critic(strata, np.random.default_rng(11), 'stratified', 4, 1)
    assert len(np.unique(ids)) == 4 and info['sampled_done'] == 1
    np.testing.assert_allclose(weights[:3], .7/.75, rtol=1e-6)
    assert weights[3] == pytest.approx(.3/.25)
    assert weights.mean() == pytest.approx(1.)
    inverse = replay_with_done([0]+[1]*7)
    ids, weights, info = sample_critic(ReplayStrata(inverse), np.random.default_rng(8), 'stratified', 4, 1)
    assert len(np.unique(ids)) == 4 and info['sampled_done'] == 3
    assert weights.mean() == pytest.approx(1.)


def test_insufficient_layer_and_empty_layer_have_valid_weights():
    m = replay_with_done([0]*7+[1])
    strata = ReplayStrata(m)
    ids, weights, info = sample_critic(strata, np.random.default_rng(7), 'stratified', 4, 2)
    assert len(np.unique(ids)) == 4 and info['sampled_done'] == 1 and not info['fallback_uniform']
    assert weights.mean() == pytest.approx(1.)
    zero = replay_with_done([0]*8)
    a = sample_critic(ReplayStrata(zero), np.random.default_rng(4), 'uniform', 4, 1)
    b = sample_critic(ReplayStrata(zero), np.random.default_rng(4), 'stratified', 4, 1)
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
    assert b[2]['fallback_uniform']


def test_replay_strata_exact_fifo_replacement_and_restore():
    m = replay_with_done([0, 1, 0, 1, 0, 1])
    strata = ReplayStrata(m)
    old = int(m.done[m.index]); index = m.index
    obs = np.zeros((3, 162), np.float32); action = np.zeros((3, 3), np.float32)
    m.push(obs, action, 0., obs, 1)
    strata.on_push(index, old, 1)
    saved = strata.state_dict()
    rebuilt = ReplayStrata(m); rebuilt.load_state_dict(saved)
    assert rebuilt.members == strata.members
    assert np.array_equal(rebuilt.position, strata.position)
    damaged = copy.deepcopy(saved)
    damaged['position'][0] = -1
    with pytest.raises(ValueError, match='disagree'):
        rebuilt.load_state_dict(damaged)


def make_checkpoint(tmp_path, arm, device='cpu'):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/masac_v3/protocol.json').read_text())
    cfg.update(steps=32, warmup=0, replay_capacity=32, smoke=False,
               arms=['standard'], seeds=[411], n_agents=3)
    (tmp_path/'protocol.json').write_text(json.dumps(cfg))
    folder = tmp_path/'standard_411'; folder.mkdir()
    torch.set_num_threads(1); torch.manual_seed(411)
    model = SharedMASAC(capacity=32, seed=411)
    env = StructuredMission(layout='urban'); obs = env.reset(511000)
    exploration = np.random.default_rng(411)
    for i in range(32):
        action = np.zeros((3,3), np.float32)
        nxt, reward, done, trunc, _ = env.step(action)
        assert not done and not trunc
        # Synthetic done labels are used only to exercise both sampler strata.
        model.push(obs, action, reward, nxt, i % 4 == 0)
        obs = nxt
    assert model.size == 32 and model.index == 0
    updater = CriticSamplingUpdater(model, arm=arm, seed=411, batch_size=16, done_quota=4, device=device)
    updater.update()
    checkpoint = capture(model, env, exploration, step=32, episode=0,
                         episode_return=0., seed=411, cfg=cfg)
    checkpoint['diagnostic_sampling'] = updater.state_dict()
    torch.save(checkpoint, folder/'checkpoint_32.pt'); save_replay(model, folder)
    return folder, cfg, model, updater, env, obs, exploration


def restore(folder, cfg, arm, device='cpu'):
    model = SharedMASAC(capacity=32, seed=411)
    cp, env, obs, exploration = restore_completed_run(model, folder, cfg)
    updater = CriticSamplingUpdater(model, arm=arm, seed=411, batch_size=16, done_quota=4, device=device)
    updater.load_state_dict(cp['diagnostic_sampling'])
    return model, updater, env, obs, exploration


def advance(model, updater, env, obs, exploration):
    outcomes = []
    for i in range(3):
        action = updater.act(obs)
        nxt, reward, done, trunc, info = env.step(action)
        assert not done and not trunc
        updater.push(obs, action, reward, nxt, bool(i == 1))
        metrics = updater.update()
        metrics = {k:v for k,v in metrics.items() if not k.endswith('_s')}
        outcomes.append((action.copy(), nxt.copy(), reward, done, trunc, info, metrics,
                         exploration.random(), model.index))
        obs = nxt
    weights = {name:{k:v.detach().cpu().clone() for k,v in getattr(model,name).state_dict().items()}
               for name in ('actor','q','target')}
    return outcomes, weights, model.logalpha.detach().cpu().clone(), updater.state_dict()


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS is unavailable'))])
@pytest.mark.parametrize('arm', ['uniform', 'stratified'])
def test_checkpoint_restore_matches_actions_updates_and_rng(tmp_path, arm, device):
    folder, cfg, model, updater, env, obs, exploration = make_checkpoint(tmp_path, arm, device)
    expected = advance(model, updater, env, obs, exploration)
    restored, restored_updater, restored_env, restored_obs, restored_exploration = restore(folder, cfg, arm, device)
    actual = advance(restored, restored_updater, restored_env, restored_obs, restored_exploration)
    for x, y in zip(expected[0], actual[0]):
        np.testing.assert_array_equal(x[0],y[0]); np.testing.assert_array_equal(x[1],y[1])
        assert x[2:] == y[2:]
    for name in expected[1]:
        for key in expected[1][name]:
            assert torch.equal(expected[1][name][key],actual[1][name][key])
    assert torch.equal(expected[2], actual[2])
    assert expected[3]['critic_rng'] == actual[3]['critic_rng']
    assert expected[3]['actor_rng'] == actual[3]['actor_rng']
    assert torch.equal(expected[3]['target_torch_rng'],actual[3]['target_torch_rng'])
    assert torch.equal(expected[3]['actor_torch_rng'],actual[3]['actor_torch_rng'])
    if device == 'mps':
        assert torch.equal(expected[3]['rollout_mps_rng'],actual[3]['rollout_mps_rng'])
    np.testing.assert_array_equal(expected[3]['strata']['position'],actual[3]['strata']['position'])


def test_arms_have_independent_but_matched_actor_rngs(tmp_path):
    folder, cfg, m, a, _, _, _ = make_checkpoint(tmp_path, 'uniform')
    cp, _, _, _ = restore_completed_run(SharedMASAC(capacity=32, seed=411), folder, cfg)
    m_b = SharedMASAC(capacity=32, seed=411)
    restore_completed_run(m_b, folder, cfg)
    b = CriticSamplingUpdater(m_b, arm='stratified', seed=411, batch_size=16, done_quota=4)
    # Reinitialize both sampling systems identically to compare the first new update.
    a = CriticSamplingUpdater(m, arm='uniform', seed=411, batch_size=16, done_quota=4)
    outer_rng=torch.random.get_rng_state().clone()
    a_metrics=a.update()
    assert torch.equal(torch.random.get_rng_state(),outer_rng)
    b_metrics=b.update()
    assert torch.equal(torch.random.get_rng_state(),outer_rng)
    assert a_metrics['actor_batch_done_fraction']==b_metrics['actor_batch_done_fraction']
    assert a.actor_rng.bit_generator.state == b.actor_rng.bit_generator.state
    assert torch.equal(a.actor_torch_rng.get_state(), b.actor_torch_rng.get_state())
    assert torch.equal(a.target_torch_rng.get_state(), b.target_torch_rng.get_state())
