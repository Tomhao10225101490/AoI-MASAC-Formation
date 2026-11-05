"""State-isolation and real process-lock contracts for formal near hooks."""
import json
import multiprocessing as mp
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from uav_planning.run_lock import RunLock
from uav_planning.v4_near_study_runtime import install_training_reward


def config(mode='potential'):
    return dict(near_objective=dict(mode=mode, coefficient=.4, radius_m=15., gamma=.99),
                sync_capture=dict(enabled=False), reshape=dict(near_goal_bonus=.4, near_goal_radius_m=15.),
                seeds=[1611, 1622], log_interval=100)


def fake_trainer(evaluate=None, update_registry=None):
    calls = []

    def audited_episode(family, cfg, reshape, seed, episode, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        env = SimpleNamespace(p=np.full((3, 3), 1.), goal=np.full((3, 3), 2.),
                              v=np.zeros((3, 3)), t=0, seed=seed)
        result = (env, object())
        calls.append(('audit', result))
        return result

    def append(path, obj):
        calls.append(('append', path, obj))
        return obj

    def write(path, obj):
        calls.append(('write', path, obj))
        return obj

    return SimpleNamespace(audited_episode=audited_episode,
                           evaluate=evaluate or (lambda *a, **k: object()),
                           shape_reward=lambda *a, **k: None, append=append, write=write,
                           update_registry=update_registry or (lambda *a, **k: None), calls=calls)


def install(tmp_path, mode='potential', trainer=None, **kwargs):
    protocol = tmp_path / 'protocol.md'
    protocol.write_text('Fixed test protocol\n', encoding='utf-8')
    out = tmp_path / 'run'
    out.mkdir(exist_ok=True)
    trainer = trainer or fake_trainer()
    runtime = install_training_reward(trainer, config(mode), out, protocol,
                                     registry_lock_path=tmp_path / 'registry.lock', **kwargs)
    return trainer, runtime, out


def audit(trainer, out, seed, episode=0):
    return trainer.audited_episode('family', {}, None, seed * 10000 + episode,
                                   episode, out / f'maneuver_static_{seed}')


def callback(trainer, reward, *, terminated=False, truncated=False):
    return trainer.shape_reward(reward, None, None, terminated=terminated, truncated=truncated,
                                gamma=.99, config=SimpleNamespace(enabled=False))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_two_seeds_multiple_episodes_terminal_timeout_and_scoped_streams(tmp_path):
    trainer, runtime, out = install(tmp_path)
    result = audit(trainer, out, 1611)
    assert result is trainer.calls[-1][1]  # Original audit result object is preserved.
    env, _ = result
    phi_reset = runtime.state['previous_phi']
    assert phi_reset > 0  # Reset geometry initializes Phi; it is not forced to zero.
    first, _ = callback(trainer, 1.)
    env.p[:] = env.goal
    terminal, _ = callback(trainer, 50., terminated=True)
    row = trainer.append(out / 'metrics.jsonl', dict(kind='episode', seed=1611, step=2))
    assert row['near_reward_steps_in_episode'] == 2
    first_rows = records(out / 'maneuver_static_1611/near_reward_transitions.jsonl')
    assert [r['training_step'] for r in first_rows] == [1, 2]
    assert first_rows[1]['following_phi_observed'] == pytest.approx(.4)
    assert first_rows[1]['following_phi_effective_for_potential'] == 0.
    assert first_rows[1]['replay_done'] is True
    assert terminal == pytest.approx(50. - .4 - first_rows[1]['previous_phi'])
    assert row['near_removed_return'] == pytest.approx(sum(r['removed_near'] for r in first_rows))
    assert row['near_potential_difference_return'] == pytest.approx(sum(r['potential_difference'] for r in first_rows))

    audit(trainer, out, 1611, episode=1)
    assert runtime.state['seed_step'] == 2
    assert runtime.state['episode_steps'] == 0
    assert runtime.state['episode_removed'] == runtime.state['episode_pbrs'] == 0.
    assert runtime.state['previous_phi'] == pytest.approx(phi_reset)
    timeout, _ = callback(trainer, 1., truncated=True)
    timeout_row = records(out / 'maneuver_static_1611/near_reward_transitions.jsonl')[-1]
    assert timeout_row['training_step'] == 3 and timeout_row['replay_done'] is False
    assert timeout_row['following_phi_effective_for_potential'] == pytest.approx(phi_reset)
    assert timeout_row['timeout_without_termination'] is True
    assert timeout == pytest.approx(1. - .01 * phi_reset - phi_reset)

    audit(trainer, out, 1622)
    assert runtime.state['seed_step'] == 0
    callback(trainer, 1.)
    second_rows = records(out / 'maneuver_static_1622/near_reward_transitions.jsonl')
    assert len(second_rows) == 1 and second_rows[0]['training_seed'] == 1622 and second_rows[0]['training_step'] == 1
    assert len(records(out / 'maneuver_static_1611/near_reward_transitions.jsonl')) == 3
    assert not (out / 'near_reward_transitions.jsonl').exists()
    assert first < 1.


def test_original_identity_evaluation_isolation_and_manifest_metadata(tmp_path):
    token = object()
    trainer = fake_trainer(evaluate=lambda *a, **k: token)
    trainer, runtime, out = install(tmp_path, mode='original', trainer=trainer)
    audit(trainer, out, 1611)
    base = np.float64(.12345678901234567)
    reward, correction = callback(trainer, base)
    assert reward is base and correction == 0.
    before = runtime.state.copy()
    assert trainer.evaluate('unchanged args', keyword='unchanged') is token
    assert runtime.state == before
    record = records(out / 'maneuver_static_1611/near_reward_transitions.jsonl')[0]
    assert record['removed_near'] == record['potential_difference'] == record['correction'] == 0.
    manifest = trainer.write(out / 'run_manifest.json', dict(original_key=1))
    assert manifest['original_key'] == 1
    assert manifest['near_objective']['mode'] == 'original'
    assert len(manifest['preregistered_near_protocol']['sha256']) == 64
    assert len(manifest['near_objective_runtime']['sha256']) == 64
    assert manifest['near_runtime_resume_supported'] is False


@pytest.mark.parametrize('target', ['p', 'v', 'previous_phi', 'seed_step'])
def test_evaluation_mutation_is_detected(tmp_path, target):
    runtime_ref = []

    def bad_evaluate():
        runtime = runtime_ref[0]
        if target in ('p', 'v'):
            getattr(runtime.state['env'], target)[0, 0] += 1
        else:
            runtime.state[target] += 1

    trainer, runtime, out = install(tmp_path, trainer=fake_trainer(evaluate=bad_evaluate))
    runtime_ref.append(runtime)
    audit(trainer, out, 1611)
    with pytest.raises(RuntimeError, match='Evaluation changed'):
        trainer.evaluate()
    assert runtime.in_evaluation is False


def test_registry_exception_releases_lock_and_wait_is_bounded(tmp_path):
    def failing_registry(*args, **kwargs):
        raise LookupError('registry writer failure')

    trainer, runtime, _ = install(tmp_path, trainer=fake_trainer(update_registry=failing_registry),
                                registry_wait_s=.06, registry_poll_s=.01)
    with pytest.raises(LookupError):
        trainer.update_registry('ignored')
    with RunLock(runtime.registry_lock_path):
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            trainer.update_registry('ignored')
        assert time.monotonic() - start < 2.
    with RunLock(runtime.registry_lock_path):
        pass


def _registry_worker(base_dir, worker_id, ready, start):
    """Spawn target deliberately performs a racy read/sleep/write under the hook."""
    base = Path(base_dir)
    registry = base / 'shared_registry.json'

    def update_registry(root, payload, primary=False):
        data = json.loads(registry.read_text(encoding='utf-8')) if registry.exists() else []
        time.sleep(.01)
        data.append(payload)
        registry.write_text(json.dumps(data), encoding='utf-8')

    trainer = fake_trainer(update_registry=update_registry)
    install_training_reward(trainer, config(), base / f'worker_{worker_id}', base / 'protocol.md',
                            registry_lock_path=base / 'shared_registry.lock',
                            registry_wait_s=10., registry_poll_s=.005)
    ready.put(worker_id)
    if not start.wait(15):
        raise TimeoutError('Test parent did not release concurrent writers')
    for index in range(8):
        trainer.update_registry(str(worker_id), dict(worker=worker_id, index=index))


def test_spawned_concurrent_registry_updates_preserve_every_record(tmp_path):
    (tmp_path / 'protocol.md').write_text('Fixed concurrency protocol\n', encoding='utf-8')
    context = mp.get_context('spawn')
    ready, start = context.Queue(), context.Event()
    processes = [context.Process(target=_registry_worker, args=(str(tmp_path), i, ready, start)) for i in range(2)]
    try:
        for process in processes:
            process.start()
        assert {ready.get(timeout=20), ready.get(timeout=20)} == {0, 1}
        start.set()
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
        data = json.loads((tmp_path / 'shared_registry.json').read_text(encoding='utf-8'))
        assert len(data) == 16
        assert {(r['worker'], r['index']) for r in data} == {(w, i) for w in range(2) for i in range(8)}
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        ready.close()
        ready.join_thread()
