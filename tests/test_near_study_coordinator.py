"""Real tmp-only fake-child scheduling tests; no scientific model or rollout."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading

import pytest

from scripts import launch_near_objective_study as coordinator


FAKE_ENTRY = r'''
import argparse
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--protocol', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--torch-threads', type=int, required=True)
args = p.parse_args()
assert args.torch_threads == 1
args.out.mkdir(parents=True, exist_ok=False)
if args.out.name != 'potential_1633':
    (args.out / 'status.json').write_text(json.dumps(dict(pid=os.getpid(), phase='fake_running')), encoding='utf-8')
(args.out / 'run_manifest.json').write_text(json.dumps(dict(pid=os.getpid())), encoding='utf-8')
print(f'FAKE_START seed={args.seed} pid={os.getpid()} protocol={args.protocol}', flush=True)
failing = (Path(__file__).parent / 'fail_first.flag').exists()
is_first = args.out.name == 'original_1611'
if failing and is_first:
    time.sleep(.05)
    print('FAKE_REGISTERED_FAILURE returncode=7 no_model=true', flush=True)
    sys.exit(7)
time.sleep(.3 if failing else .10 + (args.seed % 3) * .02)
print(f'FAKE_CHECKPOINT saved_label_only seed={args.seed} no_weights=true', flush=True)
completion = dict(complete=True, runs=[dict(seed=args.seed, step=40000)])
(args.out / 'completion.json').write_text(json.dumps(completion), encoding='utf-8')
print(f'FAKE_SAVED_EXIT seed={args.seed} complete=true', flush=True)
'''


def prepare(tmp_path, monkeypatch, fail=False):
    (tmp_path / 'fake.py').write_text(FAKE_ENTRY, encoding='utf-8')
    if fail:
        (tmp_path / 'fail_first.flag').write_text('true\n', encoding='utf-8')
    out = tmp_path / 'fake_study'
    out.mkdir()
    (out / 'runs').mkdir()
    (out / 'logs').mkdir()
    jobs = [dict(mode=mode, training_seed=seed, relative_root=f'runs/{mode}_{seed}',
                 config=f'configs/{mode}.json')
            for seed in coordinator.SEEDS for mode in ('original', 'potential')]
    frozen = dict(schema_version=1, layout=coordinator.LAYOUT, seeds=list(coordinator.SEEDS),
                  step=40000, max_workers=4, jobs=jobs, sources={}, paired_config_hashes={})
    coordinator.write(out / 'study_manifest.json', frozen, exclusive=True)
    monkeypatch.setattr(coordinator, 'ROOT', tmp_path)
    monkeypatch.setattr(coordinator, 'ENTRY', 'fake.py')
    monkeypatch.setattr(coordinator, 'check_frozen', lambda value: None)
    return out, frozen


@contextmanager
def capture_emitter(out):
    lines, lock = [], threading.Lock()
    with (out / 'coordinator.log').open('x', encoding='utf-8', buffering=1) as log:
        def emit(text):
            with lock:
                lines.append(text)
                print(text, flush=True)
                log.write(text + '\n')
                log.flush()
        yield emit, lines


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def assert_visible_launch_order_and_capacity(lines, expected_launches):
    commands = [i for i, line in enumerate(lines) if line.startswith('COMMAND BEFORE Popen:')]
    started = [i for i, line in enumerate(lines) if line.startswith('CHILD_STARTED ')]
    assert len(commands) == len(started) == expected_launches
    assert all(command < start for command, start in zip(commands, started))
    active, maximum = 0, 0
    for line in lines:
        if line.startswith('CHILD_STARTED '):
            active += 1
            maximum = max(maximum, active)
        elif line.startswith('CHILD_EXIT '):
            active -= 1
        assert 0 <= active <= 4
    assert active == 0
    assert maximum == min(4, expected_launches)


def test_six_real_fake_children_complete_with_four_slots_and_final_hashes(tmp_path, monkeypatch):
    out, frozen = prepare(tmp_path, monkeypatch)
    with capture_emitter(out) as (emit, lines):
        code = coordinator.execute(out, frozen, emit)
    assert code == 0
    assert_visible_launch_order_and_capacity(lines, expected_launches=6)
    completion, status = read(out / 'completion.json'), read(out / 'status.json')
    assert completion['complete'] is completion['all_jobs_complete'] is True
    assert status['complete'] is status['all_jobs_complete'] is True
    assert len(completion['jobs']) == 6
    assert {(job['mode'], job['training_seed']) for job in completion['jobs']} == {
        (mode, seed) for seed in coordinator.SEEDS for mode in ('original', 'potential')}
    assert all(job['returncode'] == 0 for job in completion['jobs'])
    for job in status['jobs']:
        source = 'run_manifest.json' if job['relative_root'] == 'runs/potential_1633' else 'status.json'
        actual = read(out / job['relative_root'] / source)['pid']
        assert type(job['launch_pid']) is int and job['launch_pid'] > 0
        assert job['training_pid'] == actual
        assert job['training_pid_source'] == source
    for name, expected in completion['artifact_sha256'].items():
        assert coordinator.digest(out / name) == expected
    # Completion is emitted only after every parent artifact and all child
    # checkpoint-label lines have reached their own logs.
    final_time = (out / 'completion.json').stat().st_mtime_ns
    assert all(path.stat().st_mtime_ns <= final_time for path in out.rglob('*') if path.is_file())
    for job in frozen['jobs']:
        root = out / job['relative_root']
        assert read(root / 'completion.json')['runs'] == [dict(seed=job['training_seed'], step=40000)]
        log = (out / 'logs' / f'{job["mode"]}_{job["training_seed"]}.log').read_text(encoding='utf-8')
        assert 'FAKE_CHECKPOINT saved_label_only' in log
        assert 'FAKE_SAVED_EXIT' in log
    assert not (out / 'error.json').exists()


def test_failed_first_child_halts_pending_queue_and_drains_registered_siblings(tmp_path, monkeypatch):
    out, frozen = prepare(tmp_path, monkeypatch, fail=True)
    with capture_emitter(out) as (emit, lines):
        code = coordinator.execute(out, frozen, emit)
    assert code == 1
    assert_visible_launch_order_and_capacity(lines, expected_launches=4)
    assert not (out / 'completion.json').exists()
    status, error = read(out / 'status.json'), read(out / 'error.json')
    assert status['phase'] == 'FAILED' and status['complete'] is False
    assert error['complete'] is error['all_jobs_complete'] is False
    assert error['known_running_pids'] == []
    assert len(error['failures']) == 1
    assert error['failures'][0]['job']['returncode'] == 7
    assert error['failures'][0]['job']['training_pid'] == read(out / 'runs/original_1611/status.json')['pid']
    states = {(job['mode'], job['training_seed']): job for job in error['jobs']}
    assert states[('original', 1611)]['state'] == 'FAILED'
    for mode, seed in (('potential', 1611), ('original', 1622), ('potential', 1622)):
        assert states[(mode, seed)]['state'] == 'COMPLETE'
        assert states[(mode, seed)]['returncode'] == 0
        assert states[(mode, seed)]['training_pid'] == read(out / f'runs/{mode}_{seed}/status.json')['pid']
        assert read(out / f'runs/{mode}_{seed}/completion.json')['complete'] is True
        text = (out / f'logs/{mode}_{seed}.log').read_text(encoding='utf-8')
        assert 'FAKE_CHECKPOINT saved_label_only' in text and 'FAKE_SAVED_EXIT' in text
    for mode in ('original', 'potential'):
        assert states[(mode, 1633)]['state'] == 'NOT_LAUNCHED_AFTER_FAILURE'
        assert states[(mode, 1633)]['training_pid'] is None
        assert not (out / f'runs/{mode}_1633').exists()
    assert not any('terminate' in line.lower() and 'true' in line.lower() for line in lines)


def test_cli_rejects_wrong_named_output_before_creating_it(tmp_path, monkeypatch):
    invalid = tmp_path / 'unregistered_output'
    monkeypatch.setattr(coordinator, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['launch_near_objective_study.py', '--out', str(invalid)])
    with pytest.raises(SystemExit) as caught:
        coordinator.main()
    assert caught.value.code == 2
    assert not invalid.exists()
