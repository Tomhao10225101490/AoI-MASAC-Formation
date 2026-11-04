"""Both parent formats must continue across the replay ring boundary exactly."""

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.continue_masac_v3 import capture, save_replay
from scripts.train_masac_v3_curriculum import new_episode_stats
from uav_planning.masac_v3_curriculum import make_episode
from uav_planning.shared_masac import SharedMASAC
from uav_planning.structured import StructuredMission


def make_parent(tmp_path, arm):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / 'configs/masac_v3/protocol.json').read_text())
    # A full replay whose next insertion is the final slot: continuation
    # immediately wraps around and overwrites older transitions.
    cfg.update(steps=63, warmup=0, replay_capacity=32, smoke=False,
               arms=['standard'], seeds=[411], n_agents=3)
    if arm == 'curriculum':
        cfg['stages'] = [{'name': 'full', 'start_step': 0}]
    parent = tmp_path / arm
    directory = parent / 'standard_411'
    directory.mkdir(parents=True)
    (parent / 'protocol.json').write_text(json.dumps(cfg))

    torch.set_num_threads(1)
    torch.manual_seed(411)
    np.random.seed(411)
    random.seed(411)
    exploration = np.random.default_rng(411)
    model = SharedMASAC(capacity=32, seed=411)
    if arm == 'curriculum':
        env, obs, stage = make_episode(411, 0, 0, cfg)
        assert stage == 'full'
    else:
        env = StructuredMission(layout=cfg['layouts'][0])
        obs = env.reset(cfg['training_map_seed_offset'] + 411000)
    # Preserve an in-progress physical episode; all transitions inserted in
    # replay are valid, even when duplicated to exercise a small full buffer.
    for _ in range(5):
        action = exploration.uniform(-.1, .1, (3, 3)).astype(np.float32)
        nxt, reward, done, trunc, _ = env.step(action)
        assert not done and not trunc
        model.push(obs, action, reward, nxt, done)
        obs = nxt
    for _ in range(58):
        model.push(obs, action, reward, obs, False)
    assert model.size == 32 and model.index == 31 and env.t == 5
    model.update(16)  # Ensure optimizer and target states are nontrivial.
    cp = capture(model, env, exploration, step=63, episode=0,
                 episode_return=1.25, seed=411, cfg=cfg)
    if arm == 'curriculum':
        cp.update(curriculum_stage='full', episode_start_step=0,
                  episode_stats=new_episode_stats(env), exact_resume=True)
    torch.save(cp, directory / 'checkpoint_63.pt')
    save_replay(model, directory)
    return parent, directory, model, env, obs, exploration


def advance(model, env, obs, exploration):
    steps = []
    for _ in range(3):
        action = model.act(obs)
        nxt, reward, done, trunc, info = env.step(action)
        assert not done and not trunc
        model.push(obs, action, reward, nxt, done)
        losses = model.update(16)
        draws = (random.random(), np.random.random(), exploration.random())
        steps.append((action.copy(), nxt.copy(), reward, done, trunc, info, losses,
                      draws, model.index))
        obs = nxt
    weights = {name: {k: v.clone() for k, v in getattr(model, name).state_dict().items()}
               for name in ('actor', 'q', 'target')}
    replay = {name: getattr(model, name).copy()
              for name in ('obs', 'next', 'actions', 'rewards', 'done')}
    return steps, weights, replay, model.rng.bit_generator.state, model.logalpha.detach().clone()


@pytest.mark.parametrize('arm', ['original', 'curriculum'])
def test_restored_parent_crosses_full_replay_ring_exactly(tmp_path, arm):
    from scripts.continue_masac_v3_paired import restore_parent

    parent, _, model, env, obs, rng = make_parent(tmp_path, arm)
    expected = advance(model, env, obs, rng)
    restored, restored_env, restored_obs, restored_rng, checkpoint = restore_parent(arm, parent, 411)
    assert restored.index == 31 and restored.size == 32
    assert restored_env.t == 5 and checkpoint['episode'] == 0
    np.testing.assert_array_equal(restored_obs, obs)
    actual = advance(restored, restored_env, restored_obs, restored_rng)
    for left, right in zip(expected[0], actual[0]):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2:] == right[2:]
    assert [step[-1] for step in actual[0]] == [0, 1, 2]
    for name in ('actor', 'q', 'target'):
        for key in expected[1][name]:
            assert torch.equal(expected[1][name][key], actual[1][name][key])
    for name in expected[2]:
        np.testing.assert_array_equal(expected[2][name], actual[2][name])
    assert expected[3] == actual[3]
    assert torch.equal(expected[4], actual[4])


@pytest.mark.parametrize('arm', ['original', 'curriculum'])
def test_restoration_rejects_replay_with_wrong_shape(tmp_path, arm):
    from scripts.continue_masac_v3_paired import restore_parent

    parent, directory, _, _, _, _ = make_parent(tmp_path, arm)
    with np.load(directory / 'replay_final.npz') as source:
        contents = {key: source[key] for key in source.files}
    contents['obs'] = contents['obs'][:-1]
    np.savez_compressed(directory / 'replay_final.npz', **contents)
    with pytest.raises(ValueError):
        restore_parent(arm, parent, 411)


def test_incomplete_or_errored_parent_is_rejected(tmp_path):
    from scripts.continue_masac_v3_paired import check_parent

    with pytest.raises(ValueError):
        check_parent('curriculum', tmp_path, [411], {})
    (tmp_path / 'completion.json').write_text(json.dumps({'complete': True}))
    (tmp_path / 'error.json').write_text(json.dumps({'error': 'interrupted'}))
    with pytest.raises(ValueError):
        check_parent('curriculum', tmp_path, [411], {})


def test_parent_with_changed_snapshot_is_rejected(tmp_path):
    from scripts.continue_masac_v3_paired import check_parent

    parent, _, _, _, _, _ = make_parent(tmp_path, 'original')
    cfg = json.loads((parent / 'protocol.json').read_text())
    protocol = {key: cfg[key] for key in ('seeds', 'warmup', 'batch_size', 'replay_capacity',
                                            'validation_map_seeds', 'training_map_seed_offset', 'layouts')}
    protocol['start_step'] = cfg['steps']
    (parent / 'completion.json').write_text(json.dumps({'complete': True}))
    snapshot = parent / 'source_snapshot'
    snapshot.mkdir()
    (snapshot / 'structured.py').write_text('altered source')
    (parent / 'source_hashes.json').write_text(json.dumps({'structured.py': '0' * 64}))
    with pytest.raises(ValueError, match='snapshot hash mismatch'):
        check_parent('original', parent, [411], protocol)


def test_parent_with_changed_live_source_is_rejected(tmp_path, monkeypatch):
    import scripts.continue_masac_v3_paired as paired

    parent, _, _, _, _, _ = make_parent(tmp_path, 'original')
    cfg = json.loads((parent / 'protocol.json').read_text())
    protocol = {key: cfg[key] for key in ('seeds', 'warmup', 'batch_size', 'replay_capacity',
                                            'validation_map_seeds', 'training_map_seed_offset', 'layouts')}
    protocol['start_step'] = cfg['steps']
    (parent / 'completion.json').write_text(json.dumps({'complete': True}))
    snapshot = parent / 'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for name, relpath in paired.COMMON.items():
        live = paired.ROOT / relpath
        shutil.copy2(live, snapshot / name)
        hashes[name] = paired.sha(live)
    (parent / 'source_hashes.json').write_text(json.dumps(hashes))
    real_sha = paired.sha
    live_structured = paired.ROOT / 'uav_planning/structured.py'
    monkeypatch.setattr(paired, 'sha', lambda path: '0' * 64 if Path(path) == live_structured else real_sha(path))
    with pytest.raises(ValueError, match='Active source differs from parent'):
        paired.check_parent('original', parent, [411], protocol)


def test_recovery_shard_is_reported_as_shard_not_complete_batch(tmp_path):
    from scripts.continue_masac_v3_paired import check_parent
    from scripts.masac_v3_status import describe

    (tmp_path / 'protocol.json').write_text(json.dumps({'arms': ['original'], 'seeds': [411]}))
    (tmp_path / 'status.json').write_text(json.dumps({
        'mode': 'original', 'seed': 411, 'step': 300000, 'total_steps': 300000,
        'updates': 299000, 'steps_per_s': 50., 'eta_s': 0., 'validation_success': 0.,
        'overall_total_new_steps': 100000, 'overall_completed_new_steps': 100000,
    }))
    (tmp_path / 'recovery_completion.json').write_text(json.dumps({'complete': True, 'runs': 1, 'recovery_only': True}))
    line, finished, _ = describe(tmp_path)
    assert finished and '恢复分片已完成' in line
    assert not (tmp_path / 'completion.json').exists()
    with pytest.raises(ValueError, match='complete and error-free'):
        check_parent('original', tmp_path, [411], {})


@pytest.mark.parametrize('changed', ['live_runner', 'run_snapshot', 'run_protocol', 'live_protocol'])
def test_recovery_rejects_changed_producing_code_or_protocol(tmp_path, monkeypatch, changed):
    import scripts.continue_masac_v3_paired as paired

    monkeypatch.setattr(paired, 'COMMON', {})
    monkeypatch.setattr(paired, 'CURRICULUM', {})
    source_root = tmp_path / 'workspace'
    script = source_root / 'scripts/continue_masac_v3_paired.py'
    script.parent.mkdir(parents=True)
    script.write_text('runner v1')
    protocol_file = tmp_path / 'frozen.json'
    protocol = {'start_step': 100000, 'steps': 300000}
    protocol_file.write_text(json.dumps(protocol))
    run = tmp_path / 'paired_run'
    snap = run / 'original_411' / 'snapshots' / '150000'
    snap.mkdir(parents=True)
    saved = run / 'source_snapshot'
    saved.mkdir()
    shutil.copy2(script, saved / script.name)
    shutil.copy2(protocol_file, saved / 'paired_300k_protocol.json')
    hashes = {f.name: paired.sha(f) for f in saved.iterdir()}
    (run / 'source_hashes.json').write_text(json.dumps(hashes))
    (run / 'protocol.json').write_text(json.dumps({**protocol, 'pid': 1}))
    assert paired.validate_snapshot_source(snap, protocol, protocol_file, source_root=source_root) == run
    if changed == 'live_runner':
        script.write_text('runner v2')
        match = 'Current code or paired protocol differs'
    elif changed == 'run_snapshot':
        (saved / script.name).write_text('tampered')
        match = 'Producing run snapshot hash mismatch'
    elif changed == 'run_protocol':
        (run / 'protocol.json').write_text(json.dumps({'start_step': 1, 'steps': 300000}))
        match = 'different paired protocol'
    else:
        protocol_file.write_text('changed protocol')
        match = 'Current code or paired protocol differs'
    with pytest.raises(ValueError, match=match):
        paired.validate_snapshot_source(snap, protocol, protocol_file, source_root=source_root)


def save_snapshot_fixture(tmp_path, arm):
    from scripts.continue_masac_v3_paired import save_full_snapshot

    parent, _, model, env, obs, rng = make_parent(tmp_path, arm)
    destination = tmp_path / 'extension' / arm / 'standard_411'
    destination.mkdir(parents=True)
    cfg = json.loads((parent / 'protocol.json').read_text())
    save_full_snapshot(destination, model, env, rng, step=63, episode=0,
                       episode_return=1.25, episode_start=0,
                       episode_stats=new_episode_stats(env), partial_episode=True,
                       seed=411, parent_cfg=cfg, arm=arm)
    return destination / 'snapshots' / '63', model, env, obs, rng


@pytest.mark.parametrize('arm', ['original', 'curriculum'])
def test_full_snapshot_restores_ring_and_current_episode(tmp_path, arm):
    from scripts.continue_masac_v3_paired import restore_snapshot

    snapshot, model, env, obs, rng = save_snapshot_fixture(tmp_path, arm)
    expected = advance(model, env, obs, rng)
    restored, restored_env, restored_obs, restored_rng, cp = restore_snapshot(snapshot)
    assert restored.index == 31 and restored.size == 32
    assert cp['partial_episode'] is True and restored_env.t == 5
    np.testing.assert_array_equal(restored_obs, obs)
    actual = advance(restored, restored_env, restored_obs, restored_rng)
    for left, right in zip(expected[0], actual[0]):
        np.testing.assert_array_equal(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2:] == right[2:]
    for name in ('actor', 'q', 'target'):
        for key in expected[1][name]:
            assert torch.equal(expected[1][name][key], actual[1][name][key])
    for name in expected[2]:
        np.testing.assert_array_equal(expected[2][name], actual[2][name])
    assert expected[3] == actual[3] and torch.equal(expected[4], actual[4])


@pytest.mark.parametrize('part', ['protocol.json', 'state/checkpoint_63.pt', 'state/replay_final.npz'])
def test_full_snapshot_rejects_tampered_pair(tmp_path, part):
    from scripts.continue_masac_v3_paired import restore_snapshot

    snapshot, _, _, _, _ = save_snapshot_fixture(tmp_path, 'curriculum')
    with (snapshot / part).open('ab') as f:
        f.write(b'tamper')
    with pytest.raises(ValueError, match='hash mismatch'):
        restore_snapshot(snapshot)
