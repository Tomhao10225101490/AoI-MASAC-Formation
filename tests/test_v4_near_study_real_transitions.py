"""Controlled real-physics boundary witnesses; no learner or policy evaluation."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from uav_planning.v4_near_study_runtime import install_training_reward
from uav_planning.v4_reward_reshape import make_reshape_l0_env, reshape_from_protocol

ROOT = Path(__file__).resolve().parents[1]


def test_nonzero_real_terminal_then_real_timeout_reinitializes_potential(tmp_path):
    cfg = json.loads((ROOT / 'configs/masac_v4/sync_capture_40k_control_cuda_20261105.json').read_text())
    cfg.update(seeds=[1611], near_objective=dict(mode='potential', coefficient=.4, radius_m=15., gamma=.99))
    reshape = reshape_from_protocol(cfg['reshape'])
    out = tmp_path / 'run'
    directory = out / 'maneuver_static_1611'
    directory.mkdir(parents=True)
    protocol = tmp_path / 'witness.md'
    protocol.write_text('Controlled transition witness, not a new rollout endpoint.\n')

    def audited_episode(family, cfg, reshape, seed, episode, directory):
        env, _ = make_reshape_l0_env(family, seed, sensor_range=400, reshape=reshape)
        # Existing development scene830001 has a free straight region near
        # these separated goals. Set a controlled state; no policy is run.
        env.p[:] = env.goal - np.array([3. if episode == 0 else 5., 0., 0.])
        env.v[:] = 0.
        env.prev_acc[:] = 0.
        env.t = 0 if episode == 0 else reshape.horizon - 1
        return env, env.observe()

    trainer = SimpleNamespace(audited_episode=audited_episode, evaluate=lambda *a, **k: None,
        shape_reward=lambda *a, **k: None, append=lambda *a, **k: None,
        write=lambda *a, **k: None, update_registry=lambda *a, **k: None)
    runtime = install_training_reward(trainer, cfg, out, protocol, registry_lock_path=tmp_path/'registry.lock')
    for episode in (0, 1):
        env, _ = trainer.audited_episode('urban_canyon', cfg, reshape, 830001, episode, directory)
        before = runtime.state['previous_phi']
        _, base, terminal, truncated, info = env.step(np.zeros((3, 3)))
        assert terminal is (episode == 0) and truncated is (episode == 1)
        assert all(not info[key] for key in ('uav_collision', 'static_collision', 'dynamic_collision', 'boundary_collision'))
        reward, correction = trainer.shape_reward(base, None, None, terminated=terminal,
            truncated=truncated, gamma=.99, config=SimpleNamespace(enabled=False))
        assert np.isfinite(reward) and correction < 0
        if episode == 0:
            assert info['success']
            assert before == pytest.approx(.32)
            assert correction == pytest.approx(-.64)
        else:
            assert not info['success']
            assert before == pytest.approx(.4 * (1 - 5/15))
            assert correction == pytest.approx(-1.01 * before)
    records = [json.loads(line) for line in (directory/'near_reward_transitions.jsonl').read_text().splitlines()]
    assert [r['training_step'] for r in records] == [1, 2]
    assert records[0]['following_phi_effective_for_potential'] == 0.
    assert records[0]['replay_done'] is True
    assert records[1]['replay_done'] is False and records[1]['timeout_without_termination'] is True
    assert records[1]['following_phi_effective_for_potential'] > 0
