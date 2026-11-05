#!/usr/bin/env python3
"""Engineering CUDA concurrency timing only; no scientific run or saved weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import queue
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.v4_maneuver_masac import ManeuverMASAC
from uav_planning.v4_prediction import ProbeConfig
from uav_planning.v4_reward_reshape import make_reshape_l0_env, reshape_from_protocol

CONFIG = ROOT / 'configs/masac_v4/sync_capture_40k_control_cuda_20261105.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(worker_id, count, steps, blocks, out, ready, result, start_event, start_clock):
    """Each process owns its environment, actor, critics, replay and CUDA context."""
    seed = 1800 + worker_id
    path = Path(out) / f'workers_{count}_id{worker_id}_seed{seed}.log'
    with path.open('x', encoding='utf-8', buffering=1) as log:
        def say(text):
            line = f'WORKER count={count} id={worker_id} seed={seed} {text}'
            print(line, flush=True)
            print(line, file=log, flush=True)

        try:
            torch.set_num_threads(1)
            torch.manual_seed(seed)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable')
            cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
            if cfg['sync_capture']['enabled'] or cfg['set_mode'] != 'pool':
                raise ValueError('Expected the frozen static-control pool/B-off configuration')
            reshape = reshape_from_protocol(cfg['reshape'])
            probe = ProbeConfig(world=tuple(cfg.get('world', [240., 240., 90.])),
                                body_radius=float(cfg.get('body_radius', 1.)),
                                safety_margin=float(cfg.get('safety_margin', 0.)),
                                dt=float(cfg.get('dt', .2)), horizon=float(cfg.get('probe_horizon_s', 2.)),
                                clearance_scale=float(cfg.get('clearance_scale', 40.)))
            family, map_seed = 'urban_canyon', 830001
            env, obs = make_reshape_l0_env(family, map_seed, sensor_range=cfg['sensor_range'], reshape=reshape)
            model = ManeuverMASAC(n=3, capacity=cfg['replay_capacity'], seed=seed,
                                 device='cuda', set_mode='pool', probe_mode='static', probe_config=probe)
            rng = np.random.default_rng(seed)
            counts = dict(prefill_transitions=0, warmup_updates=0, measured_updates=0,
                          finite_action_values=0, finite_observation_values=0,
                          finite_rewards=0, finite_diagnostic_values=0, nonfinite_values=0)

            def finite(name, value):
                array = np.asarray(value)
                bad = int((~np.isfinite(array)).sum())
                counts['nonfinite_values'] += bad
                if bad:
                    raise FloatingPointError(f'{name}: {bad} nonfinite values')
                return int(array.size)

            def transition(action, update):
                nonlocal env, obs
                counts['finite_action_values'] += finite('action', action)
                next_obs, reward, done, truncated, _ = env.step(action)
                counts['finite_observation_values'] += finite('observation', next_obs)
                counts['finite_rewards'] += finite('reward', reward)
                model.push(obs, action, reward, next_obs, done)
                metrics = model.update(batch=128) if update else {}
                for name, value in metrics.items():
                    counts['finite_diagnostic_values'] += finite(f'diagnostic {name}', value)
                obs = next_obs
                if done or truncated:
                    env, obs = make_reshape_l0_env(family, map_seed, sensor_range=cfg['sensor_range'], reshape=reshape)

            say(f'START device=cuda torch_threads=1 probe=static B=false batch=128 '
                f'capacity={cfg["replay_capacity"]} prefill=160 warmup_updates=8 '
                f'steps_per_block={steps} blocks={blocks} map={family}/{map_seed} '
                'scope=engineering_only weights_saved=false scientific_endpoint=false')
            for _ in range(160):
                transition(rng.uniform(-1, 1, (3, 3)).astype(np.float32), False)
                counts['prefill_transitions'] += 1
            for _ in range(8):
                transition(model.act(obs), True)
                counts['warmup_updates'] += 1
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ready.put(dict(worker_id=worker_id, seed=seed))
            say('READY waiting_for_common_start_event=true')
            if not start_event.wait(240):
                raise TimeoutError('Parent did not release the common start event')
            common_start = start_clock.value
            block_results = []
            for block in range(blocks):
                torch.cuda.synchronize()
                began = time.perf_counter()
                for index in range(steps):
                    transition(model.act(obs), True)
                    counts['measured_updates'] += 1
                    if (index + 1) % 64 == 0 or index + 1 == steps:
                        say(f'PROGRESS block={block + 1}/{blocks} ticks={index + 1}/{steps} '
                            f'measured_updates={counts["measured_updates"]} '
                            f'common_elapsed_s={time.perf_counter() - common_start:.3f} finite=true')
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                block_results.append(dict(block=block + 1, steps=steps, elapsed_s=elapsed,
                                          steps_per_s=steps / elapsed))
                say(f'BLOCK_COMPLETE block={block + 1}/{blocks} elapsed_s={elapsed:.6f} '
                    f'steps_per_s={steps / elapsed:.6f}')
            torch.cuda.synchronize()
            finished = time.perf_counter()
            row = dict(worker_id=worker_id, seed=seed, pid=mp.current_process().pid,
                       measured_steps=steps * blocks, common_elapsed_s=finished - common_start,
                       worker_steps_per_s=steps * blocks / (finished - common_start), blocks=block_results,
                       cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_max_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                       finite_counts=counts, torch_threads=torch.get_num_threads(),
                       cache_risk=bool(model.cache_risk), parameter_counts=model.parameter_counts(),
                       gpu=torch.cuda.get_device_name(0), log=str(path.resolve()))
            if not row['cache_risk']:
                raise ValueError('Benchmark unexpectedly disabled cached replay risk')
            say(f'FINISH measured_steps={steps * blocks} common_elapsed_s={row["common_elapsed_s"]:.6f} '
                f'worker_steps_per_s={row["worker_steps_per_s"]:.6f} '
                f'cuda_max_allocated={row["cuda_max_memory_allocated_bytes"]} finite_counts={counts}')
            result.put(dict(kind='result', row=row))
        except BaseException:
            stack = traceback.format_exc()
            say(f'ERROR\n{stack}')
            result.put(dict(kind='error', worker_id=worker_id, seed=seed, traceback=stack))
            raise


def nvidia_sample():
    command = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw',
               '--format=csv,noheader,nounits']
    print('COMMAND BEFORE subprocess: ' + subprocess.list2cmdline(command), flush=True)
    began = time.perf_counter()
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
        record = dict(command=command, returncode=completed.returncode, stdout=completed.stdout.strip(),
                      stderr=completed.stderr.strip(), scope='One point sample per case, not continuous utilization')
    except (OSError, subprocess.TimeoutExpired) as exc:
        record = dict(command=command, error=repr(exc), scope='Single point sample unavailable')
    record['command_elapsed_s'] = time.perf_counter() - began
    print(f'NVIDIA_SINGLE_POINT_SAMPLE {record}', flush=True)
    return record


def run_case(context, count, args):
    ready, result = context.Queue(), context.Queue()
    start_event, start_clock = context.Event(), context.Value('d', 0.)
    processes = []
    rows = {}
    deadline = time.perf_counter() + args.timeout_s
    last_heartbeat = time.perf_counter()

    def heartbeat(phase, ready_count):
        nonlocal last_heartbeat
        now = time.perf_counter()
        if now - last_heartbeat >= 10:
            print(f'PARENT_PROGRESS workers={count} phase={phase} ready={ready_count}/{count} '
                  f'finished={len(rows)}/{count} alive={[p.pid for p in processes if p.is_alive()]} '
                  f'timeout_remaining_s={max(0., deadline - now):.1f}', flush=True)
            last_heartbeat = now
        if now >= deadline:
            raise TimeoutError(f'Case {count} exceeded {args.timeout_s}s')

    def consume_result():
        while True:
            try:
                message = result.get_nowait()
            except queue.Empty:
                break
            if message['kind'] == 'error':
                raise RuntimeError(f'Worker failed: {message}')
            row = message['row']
            if row['worker_id'] in rows:
                raise RuntimeError('Duplicate worker result')
            rows[row['worker_id']] = row

    try:
        for worker_id in range(count):
            print(f'WORKER BEFORE spawn count={count} id={worker_id} seed={1800 + worker_id} '
                  f'device=cuda torch_threads=1 probe=static batch=128 prefill=160 warmup=8 '
                  f'steps={args.steps} blocks={args.blocks} out={args.out.resolve()}', flush=True)
            process = context.Process(target=worker, args=(worker_id, count, args.steps, args.blocks,
                                      str(args.out.resolve()), ready, result, start_event, start_clock))
            process.start()
            processes.append(process)
        ready_ids = set()
        while len(ready_ids) < count:
            consume_result()
            try:
                message = ready.get(timeout=.5)
                ready_ids.add(message['worker_id'])
            except queue.Empty:
                pass
            if any(p.exitcode is not None and p.exitcode != 0 for p in processes):
                consume_result()
                raise RuntimeError('A worker exited before the common start')
            heartbeat('prefill_and_warmup', len(ready_ids))
        with start_clock.get_lock():
            start_clock.value = time.perf_counter()
        print(f'CASE_COMMON_START workers={count} common_perf_counter={start_clock.value:.9f}', flush=True)
        start_event.set()
        gpu_sample = nvidia_sample()
        while len(rows) < count:
            consume_result()
            if len(rows) == count:
                break
            if any(p.exitcode is not None and p.exitcode != 0 for p in processes):
                consume_result()
                raise RuntimeError('A measured worker exited with an error')
            heartbeat('measured', len(ready_ids))
            time.sleep(.2)
        for process in processes:
            while process.is_alive():
                process.join(timeout=.5)
                heartbeat('joining_completed_workers', len(ready_ids))
            if process.exitcode != 0:
                raise RuntimeError(f'Worker pid={process.pid} exitcode={process.exitcode}')
        ordered = [rows[i] for i in range(count)]
        outer_elapsed = max(row['common_elapsed_s'] for row in ordered)
        total = sum(row['measured_steps'] for row in ordered)
        record = dict(workers=count, total_measured_steps=total,
                      outer_common_start_to_last_worker_finish_s=outer_elapsed,
                      aggregate_steps_per_s=total / outer_elapsed, worker_results=ordered,
                      gpu_single_point_sample=gpu_sample,
                      timing='Common parent perf_counter before Event.set to last CUDA-synchronized worker finish; excludes initialization/prefill/warmup/process teardown; includes block/progress logging.')
        print(f'CASE_COMPLETE workers={count} total_steps={total} outer_elapsed_s={outer_elapsed:.6f} '
              f'aggregate_steps_per_s={record["aggregate_steps_per_s"]:.6f}', flush=True)
        save(args.out / f'case_{count}.json', record)
        return record
    finally:
        # Terminate only children created for this engineering case, never any
        # other Python/training process on the machine.
        start_event.set()
        for process in processes:
            if process.is_alive():
                print(f'BENCH_CHILD_CLEANUP pid={process.pid} terminate=true', flush=True)
                process.terminate()
            process.join(timeout=5)
        ready.close()
        result.close()
        ready.join_thread()
        result.join_thread()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--blocks', type=int, default=2)
    parser.add_argument('--timeout-s', type=float, default=300.)
    args = parser.parse_args()
    if args.workers != sorted(set(args.workers)) or not all(n in (2, 3, 4) for n in args.workers):
        parser.error('--workers must be an ascending unique subset of 2 3 4')
    if args.steps < 1 or args.blocks < 1 or not np.isfinite(args.timeout_s) or args.timeout_s < 1:
        parser.error('Steps, blocks and finite timeout must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    record = dict(scope='Engineering throughput only, not learning evidence, scientific extra seeds or efficacy endpoints.',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config=str(CONFIG.relative_to(ROOT)), config_sha256=sha(CONFIG), script_sha256=sha(__file__),
                  torch=str(torch.__version__), numpy=np.__version__, python=sys.version, platform=platform.platform(),
                  cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  workload='Static probes/pool, stochastic actor + NumPy environment + cached replay + batch128 SAC update, one Torch thread, CUDA, frozen control/B-off environment reward.',
                  sampling='Worker seed1800+id matched across cases; prefill160+warmup8, then steps*blocks per worker; no evaluations, reference-map audits, snapshots, learned weights or scientific output directories.',
                  limitations='One run per concurrency in listed order; thermal/order variation and synthetic early replay population limit precision. GPU point sample is not sustained utilization.',
                  results=[])
    save(args.out / 'manifest.json', {k: v for k, v in record.items() if k != 'results'})
    try:
        context = mp.get_context('spawn')
        for count in args.workers:
            print(f'CASE_PREPARE workers={count} measured_steps_per_worker={args.steps * args.blocks}', flush=True)
            record['results'].append(run_case(context, count, args))
        save(args.out / 'benchmark.json', record)
        save(args.out / 'completion.json', dict(complete=True, cases=args.workers,
                                               benchmark_sha256=sha(args.out / 'benchmark.json'),
                                               scientific_training=False))
        print(f'CONCURRENCY_BENCH_COMPLETE output={args.out.resolve()} '
              f'aggregate_steps_per_s={[(r["workers"], r["aggregate_steps_per_s"]) for r in record["results"]]}', flush=True)
    except BaseException:
        stack = traceback.format_exc()
        print(stack, file=sys.stderr, flush=True)
        save(args.out / 'error.json', dict(complete=False, traceback=stack, completed_cases=record['results']))
        raise


if __name__ == '__main__':
    main()
