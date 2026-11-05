#!/usr/bin/env python3
"""Launch exactly six registered cold histories in at most four visible slots.

No retry, resume, seed replacement, intermediate-result selection or process
termination is performed. A failed child stops the pending queue; already
running registered siblings are drained to their normal saved exit. Ctrl-C
also stops the queue and records known PIDs while continuing that safe drain.
This coordinator never trains, evaluates or loads a checkpoint itself.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SEEDS, MAX_WORKERS, STEP = (1611, 1622, 1633), 4, 40000
LAYOUT = 'near-objective-independent-histories-v1'
ENTRY = 'scripts/train_masac_v4_near_objective_study.py'
PROTOCOL = 'docs/design/MASAC_V4_NEAR_OBJECTIVE_40K_PROTOCOL_20261105.md'
ANALYZER = 'scripts/analyze_near_objective_study.py'
COORDINATOR = 'scripts/launch_near_objective_study.py'
BASE_SOURCES = (
    'scripts/train_masac_v4_maneuver.py', 'uav_planning/v4_maneuver_masac.py',
    'uav_planning/v4_networks.py', 'uav_planning/v4_prediction.py',
    'uav_planning/v4_reward_reshape.py', 'uav_planning/v4_training.py',
    'uav_planning/v4_environment.py', 'uav_planning/v4_maps.py',
    'uav_planning/shared_masac.py', 'uav_planning/structured.py',
    'uav_planning/environment.py', 'uav_planning/v4_reference.py',
    'uav_planning/run_lock.py', 'uav_planning/v4_sync_capture.py',
)
SOURCES = (*BASE_SOURCES, 'uav_planning/v4_near_objective.py',
           'uav_planning/v4_near_study_runtime.py', ENTRY, PROTOCOL, ANALYZER, COORDINATOR)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj, *, exclusive=False):
    path = Path(path)
    text = json.dumps(obj, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        return
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def config_name(mode):
    return f'configs/masac_v4/near_objective_40k_{mode}_cuda_20261105.json'


def normalized(cfg):
    value = copy.deepcopy(cfg)
    for key in ('experiment_id', 'suite_experiment_id', 'change_set'):
        value.pop(key)
    value['near_objective']['mode'] = 'MATCHED_MODE_PLACEHOLDER'
    return value


def freeze_inputs(out, evidence):
    configs = {mode: read(ROOT / config_name(mode)) for mode in ('original', 'potential')}
    for mode, cfg in configs.items():
        if (cfg['seeds'] != list(SEEDS) or cfg['steps'] != STEP or cfg['device'] != 'cuda' or
                cfg['sync_capture']['enabled'] is not False or cfg['near_objective'] != dict(
                    mode=mode, coefficient=.4, radius_m=15., gamma=.99)):
            raise ValueError(f'Unexpected registered configuration: {mode}')
    if normalized(configs['original']) != normalized(configs['potential']):
        raise ValueError('Paired configs differ beyond mode and nominal identifiers')
    legacy = read(ROOT / 'results/masac_v4_sync_40k_control_cuda_20261105/run_manifest.json')
    if set(legacy['sources']) != set(BASE_SOURCES):
        raise ValueError('Original frozen source set is not the named fourteen files')
    if any(digest(ROOT / name) != expected for name, expected in legacy['sources'].items()):
        raise ValueError('Original fourteen sources changed')
    sources = {name: digest(ROOT / name) for name in SOURCES}
    if len(sources) != 20:
        raise ValueError('Expected exactly twenty unique declared sources')
    paired_hashes = {config_name(mode): digest(ROOT / config_name(mode)) for mode in configs}
    evidence = evidence.resolve()
    bench = read(evidence)
    cases = {case['workers']: case for case in bench['results']}
    if set(cases) != {2, 3, 4} or any(cases[n]['aggregate_steps_per_s'] <= 0 for n in cases):
        raise ValueError('Benchmark must contain the completed positive 2/3/4-worker measurements')
    for case in cases.values():
        if len(case['worker_results']) != case['workers'] or any(
                worker['finite_counts']['nonfinite_values'] != 0 for worker in case['worker_results']):
            raise ValueError('Incomplete or nonfinite concurrency evidence')
    jobs = [dict(mode=mode, training_seed=seed, relative_root=f'runs/{mode}_{seed}',
                 config=config_name(mode)) for seed in SEEDS for mode in ('original', 'potential')]
    frozen = dict(schema_version=1, layout=LAYOUT, created_at_utc=stamp(),
                  seeds=list(SEEDS), step=STEP, max_workers=MAX_WORKERS, jobs=jobs,
                  sources=sources, paired_config_hashes=paired_hashes,
                  benchmark_evidence=dict(path=str(evidence), sha256=digest(evidence),
                      scope='External engineering evidence only; no scientific history or efficacy endpoint',
                      aggregate_steps_per_s={str(n): cases[n]['aggregate_steps_per_s'] for n in (2, 3, 4)}),
                  python_executable=sys.executable, torch_threads_per_child=1,
                  total_training_transitions=6 * STEP, retries=0, resume=False,
                  failure_action='Halt pending queue, let already-running registered siblings finish/save; no automatic restart',
                  child_completion_scope='Returncode zero plus own complete=true; scientific decision is a separate frozen analyzer')
    write(out / 'study_manifest.json', frozen, exclusive=True)
    for name in (*sources, *paired_hashes):
        target = out / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        expected = sources.get(name, paired_hashes.get(name))
        if digest(target) != expected:
            raise ValueError(f'Frozen snapshot copy mismatch: {name}')
    return frozen


def check_frozen(frozen):
    for name, expected in {**frozen['sources'], **frozen['paired_config_hashes']}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen launch input changed: {name}')
    evidence = frozen['benchmark_evidence']
    if digest(evidence['path']) != evidence['sha256']:
        raise ValueError('Concurrency benchmark evidence changed')


def optional_record(path):
    if not path.exists():
        return None
    try:
        return read(path)
    except (OSError, ValueError) as exc:
        return dict(read_error=repr(exc), path=str(path))


def execute(out, frozen, emit):
    jobs = [dict(job, state='PENDING', launch_pid=None, training_pid=None,
                 training_pid_source=None, returncode=None) for job in frozen['jobs']]
    active, failures, reader_errors = {}, [], queue.Queue()
    halted, interrupted = False, False
    next_job, last_status = 0, 0.

    def job_root(job):
        return out / job['relative_root']

    def view():
        rows = []
        for job in jobs:
            root = job_root(job)
            child_status = optional_record(root / 'status.json')
            child_manifest = optional_record(root / 'run_manifest.json')
            training_pid, source = None, None
            for record, name in ((child_status, 'status.json'), (child_manifest, 'run_manifest.json')):
                candidate = record.get('pid') if isinstance(record, dict) else None
                if type(candidate) is int and candidate > 0:
                    training_pid, source = candidate, name
                    break
            # Windows venv python.exe may be a launcher/redirector process.
            # Only a child's own status/manifest identifies its interpreter PID.
            job.update(training_pid=training_pid, training_pid_source=source)
            rows.append(dict(job, child_status=child_status,
                             child_manifest_pid=child_manifest.get('pid') if isinstance(child_manifest, dict) else None,
                             child_completion=optional_record(root / 'completion.json'),
                             child_error=optional_record(root / 'error.json')))
        return rows

    def running_pids():
        view()  # Refresh actual interpreter PIDs without inferring them from launchers.
        launch = [v['process'].pid for v in active.values()]
        training = sorted({jobs[index]['training_pid'] for index in active
                           if jobs[index]['training_pid'] is not None})
        return dict(running_launch_pids=launch, running_training_pids=training,
                    running_pids=sorted(set(launch) | set(training)))

    def save_status(force=False):
        nonlocal last_status
        now = time.monotonic()
        if not force and now - last_status < 5:
            return
        phase = ('INTERRUPTED_DRAINING' if interrupted else 'FAILED_DRAINING') if halted and active else (
                'FAILED' if halted else 'RUNNING')
        write(out / 'status.json', dict(schema_version=1, phase=phase, complete=False,
              updated_at_utc=stamp(), coordinator_pid=os.getpid(), **running_pids(),
              completed_jobs=sum(job['state'] == 'COMPLETE' for job in jobs), total_jobs=6,
              no_automatic_retry=True, jobs=view()))
        emit(f'COORDINATOR_PROGRESS phase={phase} complete={sum(j["state"] == "COMPLETE" for j in jobs)}/6 '
             f'active={[(jobs[i]["mode"], jobs[i]["training_seed"], v["process"].pid, jobs[i]["training_pid"]) for i, v in active.items()]} '
             f'pending={sum(j["state"] == "PENDING" for j in jobs)} failures={len(failures)}')
        last_status = now

    def record_failure(reason, job=None, stack=None):
        nonlocal halted
        halted = True
        pids = running_pids()
        failures.append(dict(time_utc=stamp(), reason=reason,
                             job={k: job[k] for k in ('mode', 'training_seed', 'relative_root', 'launch_pid',
                                                     'training_pid', 'training_pid_source', 'returncode')} if job else None,
                             traceback=stack))
        write(out / 'error.json', dict(complete=False, all_jobs_complete=False, updated_at_utc=stamp(),
              no_automatic_retry=True, failures=failures, jobs=view(),
              known_running_pids=pids['running_pids'],
              known_running_launch_pids=pids['running_launch_pids'],
              known_running_training_pids=pids['running_training_pids'],
              handling='Pending queue halted; registered active children are not killed and their stdout is still drained.'))
        emit(f'COORDINATOR_FAILURE reason={reason} no_new_jobs=true known_running_pids='
             f'{pids["running_pids"]} launch_pids={pids["running_launch_pids"]} '
             f'training_pids={pids["running_training_pids"]}')

    def reader(process, job, log_path):
        try:
            with log_path.open('x', encoding='utf-8', buffering=1) as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    emit(f'[{job["mode"]}_{job["training_seed"]}] {line.rstrip()}')
        except BaseException:
            reader_errors.put(dict(job_index=jobs.index(job), traceback=traceback.format_exc()))
        finally:
            if process.stdout is not None:
                process.stdout.close()

    def spawn(index):
        job = jobs[index]
        check_frozen(frozen)
        root = job_root(job)
        if root.exists():
            raise FileExistsError(f'Child output must not exist before launch: {root}')
        command = [sys.executable, '-u', str(ROOT / ENTRY), '--protocol', str(ROOT / job['config']),
                   '--out', str(root), '--seed', str(job['training_seed']), '--torch-threads', '1']
        job['command'] = command
        emit('COMMAND BEFORE Popen: ' + subprocess.list2cmdline(command))
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1, env=environment,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            start_new_session=os.name != 'nt')
        job.update(state='RUNNING', launch_pid=process.pid, started_at_utc=stamp())
        thread = threading.Thread(target=reader, args=(process, job,
                 out / 'logs' / f'{job["mode"]}_{job["training_seed"]}.log'), daemon=True)
        active[index] = dict(process=process, reader=thread)
        thread.start()
        emit(f'CHILD_STARTED mode={job["mode"]} seed={job["training_seed"]} launch_pid={process.pid} training_pid=null '
             f'ownout={root} max_workers={MAX_WORKERS}')

    save_status(force=True)
    while active or (not halted and next_job < len(jobs)):
        try:
            while not reader_errors.empty():
                error = reader_errors.get_nowait()
                record_failure('Child stdout reader failed', jobs[error['job_index']], error['traceback'])
            for index in list(active):
                process = active[index]['process']
                code = process.poll()
                if code is None:
                    continue
                active[index]['reader'].join(timeout=.2)
                if active[index]['reader'].is_alive():
                    continue  # Drain every final checkpoint/error line before reclaiming the slot.
                job = jobs[index]
                job.update(returncode=code, finished_at_utc=stamp())
                view()
                completion = optional_record(job_root(job) / 'completion.json')
                good = code == 0 and isinstance(completion, dict) and completion.get('complete') is True
                if good:
                    runs = completion.get('runs')
                    good = (isinstance(runs, list) and len(runs) == 1 and
                            runs[0].get('seed') == job['training_seed'] and runs[0].get('step') == STEP)
                job['state'] = 'COMPLETE' if good else 'FAILED'
                del active[index]
                emit(f'CHILD_EXIT mode={job["mode"]} seed={job["training_seed"]} launch_pid={process.pid} '
                     f'training_pid={job["training_pid"]} '
                     f'returncode={code} own_final_completion_valid={good}')
                if not good:
                    record_failure('Nonzero child exit or missing/invalid own final completion', job)
            while not halted and len(active) < MAX_WORKERS and next_job < len(jobs):
                try:
                    spawn(next_job)
                    next_job += 1
                except Exception:
                    jobs[next_job]['state'] = 'FAILED_TO_START'
                    record_failure('Child preflight or Popen failed', jobs[next_job], traceback.format_exc())
            save_status()
            if active:
                time.sleep(.2)
        except KeyboardInterrupt:
            interrupted = True
            record_failure('Coordinator interrupted; no implicit resume or new launches', stack=traceback.format_exc())
            save_status(force=True)
            emit('INTERRUPTED: draining already-running registered children to saved exit; '
                 'no child or unrelated process is terminated. Known PIDs are recorded in error.json.')
        except Exception:
            record_failure('Coordinator monitoring failed', stack=traceback.format_exc())
            save_status(force=True)
            time.sleep(.5)
    while not reader_errors.empty():
        error = reader_errors.get_nowait()
        record_failure('Child stdout reader failed', jobs[error['job_index']], error['traceback'])
    if halted or len(jobs) != 6 or any(job['state'] != 'COMPLETE' for job in jobs):
        for job in jobs:
            if job['state'] == 'PENDING':
                job['state'] = 'NOT_LAUNCHED_AFTER_FAILURE'
        write(out / 'error.json', dict(complete=False, all_jobs_complete=False, failures=failures,
              jobs=view(), known_running_pids=[], known_running_launch_pids=[],
              known_running_training_pids=[], no_automatic_retry=True, updated_at_utc=stamp()))
        write(out / 'status.json', dict(phase='FAILED', complete=False, all_jobs_complete=False,
              total_jobs=6, jobs=view(), known_running_pids=[], known_running_launch_pids=[],
              known_running_training_pids=[], updated_at_utc=stamp()))
        emit('COORDINATOR_FAILED all_jobs_complete=false; original outputs retained; no retry was launched.')
        return 1
    check_frozen(frozen)
    status = dict(phase='COMPLETE', complete=True, all_jobs_complete=True,
                  completed_jobs=6, total_jobs=6, jobs=view(), updated_at_utc=stamp())
    write(out / 'status.json', status)
    emit('COORDINATOR_COMPLETE all_jobs_complete=true histories=6 steps_per_history=40000 '
         'scientific_decision=separate_frozen_analyzer')
    artifact_hashes = {name: digest(out / name) for name in ('study_manifest.json', 'status.json', 'coordinator.log')}
    artifact_hashes.update({f'logs/{job["mode"]}_{job["training_seed"]}.log':
        digest(out / 'logs' / f'{job["mode"]}_{job["training_seed"]}.log') for job in jobs})
    write(out / 'completion.json', dict(complete=True, all_jobs_complete=True, schema_version=1,
          layout=LAYOUT, created_at_utc=stamp(), jobs=[{key: job[key] for key in
             ('mode', 'training_seed', 'relative_root', 'returncode')} for job in jobs],
          artifact_sha256=artifact_hashes, no_automatic_retry=True, efficacy_decision=None), exclusive=True)
    return 0  # completion.json is the last artifact write.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--benchmark-evidence', type=Path,
                        default=ROOT / 'results/masac_v4_near_cuda_concurrency_20261105/benchmark.json')
    args = parser.parse_args()
    out = args.out.resolve()
    expected_out = (ROOT / 'results/masac_v4_near_objective_40k_sharded_cuda_20261105').resolve()
    if out != expected_out:
        parser.error(f'This registered study requires the fixed output directory: {expected_out}')
    out.mkdir(parents=True, exist_ok=False)  # Existing evidence is never modified.
    (out / 'runs').mkdir()
    (out / 'logs').mkdir()
    output_lock = threading.Lock()
    with (out / 'coordinator.log').open('x', encoding='utf-8', buffering=1) as log:
        def emit(text):
            with output_lock:
                print(text, flush=True)
                print(text, file=log, flush=True)

        try:
            emit(f'COORDINATOR_PREFLIGHT out={out} max_workers=4 fixed_histories=6 '
                 'cold_start=true resume=false retries=0 no_new_scientific_budget=true')
            frozen = freeze_inputs(out, args.benchmark_evidence)
            emit(f'STUDY_INPUTS_FROZEN manifest_sha256={digest(out / "study_manifest.json")} '
                 f'sources={len(frozen["sources"])} paired_config_hashes={frozen["paired_config_hashes"]}')
            code = execute(out, frozen, emit)
        except BaseException:
            stack = traceback.format_exc()
            emit(stack)
            if not (out / 'error.json').exists():
                write(out / 'error.json', dict(complete=False, traceback=stack, no_automatic_retry=True,
                                             known_running_pids=[], failed_at_utc=stamp()))
            write(out / 'status.json', dict(phase='FAILED', complete=False, traceback=stack))
            code = 1
    raise SystemExit(code)


if __name__ == '__main__':
    main()
