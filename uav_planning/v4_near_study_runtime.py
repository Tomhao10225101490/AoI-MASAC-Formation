"""Explicit training-only hooks for a cold-start near-objective study.

Entry points call ``install_training_reward(trainer, cfg, out, protocol_path)``
once, before ``trainer.run``. The returned runtime exposes ``state`` for read-only
inspection. The entry point must explicitly append this module, the pure near
adapter, its own entry point/protocol/analyzer to ``trainer.SOURCES``; this module
does not mutate the frozen source list. No environment, learner, or RNG is created.

Each audited training directory must be ``out/maneuver_static_{configured_seed}``.
Episode zero starts each new seed; per-seed steps restart at zero, whereas every
episode initializes its potential from its actual reset geometry. Evaluation is
delegated unchanged and checked against a snapshot of the training state.

The registry lock serializes only participating callers of update_registry. It
does not make unrelated legacy writers lock-aware, and does not imply resumable
runtime state or exact CUDA continuation.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import time

import numpy as np

from . import v4_near_objective as near_math
from .run_lock import RunLock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_LOCK = ROOT / 'results/locks/near_study_registry.lock'
SEMANTICS = 'near replacement correction, not B shaping; base_return remains original Path-C'


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _duration(value, name, positive=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


class NearStudyRuntime:
    def __init__(self, trainer, cfg, out, protocol_path, *, registry_lock_path=None,
                 registry_wait_s=30.0, registry_poll_s=0.05):
        self.trainer, self.out = trainer, Path(out).resolve()
        self.protocol = Path(protocol_path)
        self.protocol = (ROOT / self.protocol).resolve() if not self.protocol.is_absolute() else self.protocol.resolve()
        if not self.protocol.is_file():
            raise ValueError(f'Missing near-study protocol: {self.protocol}')
        self.near = dict(cfg['near_objective'])
        self.mode = self.near.get('mode')
        if self.mode not in ('original', 'potential') or self.near != dict(
                mode=self.mode, coefficient=.4, radius_m=15., gamma=.99):
            raise ValueError('Near study requires the fixed mode/coefficient/radius/gamma intervention')
        if cfg['sync_capture']['enabled'] is not False:
            raise ValueError('Both near-objective arms require B disabled')
        if (cfg['reshape']['near_goal_bonus'] != .4 or cfg['reshape']['near_goal_radius_m'] != 15.):
            raise ValueError('Removed near term must match the fixed Path-C coefficient/radius')
        seeds = cfg['seeds']
        if (not isinstance(seeds, list) or not seeds or
                any(isinstance(s, bool) or not isinstance(s, Integral) for s in seeds) or
                len(set(seeds)) != len(seeds)):
            raise ValueError('Configured training seeds must be a nonempty list of unique integers')
        # The formal entry point enforces [1611,1622,1633]. Keeping this helper
        # independent of the run budget permits small fake-trainer tests.
        self.seeds = tuple(int(s) for s in seeds)
        self.log_interval = cfg.get('log_interval', 100)
        if isinstance(self.log_interval, bool) or not isinstance(self.log_interval, Integral) or self.log_interval <= 0:
            raise ValueError('log_interval must be a positive integer')
        self.registry_lock_path = Path(registry_lock_path or DEFAULT_REGISTRY_LOCK).resolve()
        self.registry_wait_s = _duration(registry_wait_s, 'registry_wait_s')
        self.registry_poll_s = _duration(registry_poll_s, 'registry_poll_s', positive=True)
        self.protocol_sha256 = _digest(self.protocol)
        self.runtime_sha256 = _digest(__file__)
        self.adapter_sha256 = _digest(near_math.__file__)
        self.original = {name: getattr(trainer, name) for name in
                         ('audited_episode', 'evaluate', 'shape_reward', 'append', 'write', 'update_registry')}
        if not all(callable(value) for value in self.original.values()):
            raise ValueError('Trainer hooks must all be callable')
        self.audit_signature = inspect.signature(self.original['audited_episode'])
        self.state = dict(env=None, directory=None, seed=None, episode=None, map_seed=None,
                          previous_phi=None, seed_step=0, episode_steps=0,
                          episode_removed=0., episode_pbrs=0.)
        self.seen_seeds = set()
        self.in_evaluation = False

    def _phi(self, env):
        return near_math.near_potential(env.p, env.goal, coefficient=.4, radius_m=15.)

    def audited_episode(self, *args, **kwargs):
        if self.in_evaluation:
            raise RuntimeError('Evaluation attempted to invoke the training episode hook')
        bound = self.audit_signature.bind(*args, **kwargs).arguments
        directory = Path(bound['directory']).resolve()
        match = re.fullmatch(r'maneuver_static_([0-9]+)', directory.name)
        if not match or directory.parent != self.out:
            raise ValueError(f'Training directory does not identify an owned static seed: {directory}')
        seed = int(match.group(1))
        episode, map_seed = bound['episode'], bound['seed']
        if (seed not in self.seeds or isinstance(episode, bool) or not isinstance(episode, Integral)
                or episode < 0 or isinstance(map_seed, bool) or not isinstance(map_seed, Integral)):
            raise ValueError('Invalid configured training seed, episode, or map seed')
        new_seed = seed != self.state['seed']
        if new_seed:
            if episode != 0 or seed in self.seen_seeds:
                raise ValueError('Cold study requires episode zero on each new seed and forbids seed re-entry')
        elif episode != self.state['episode'] + 1:
            raise ValueError('Training episodes must advance exactly once within a seed')
        result = self.original['audited_episode'](*args, **kwargs)
        env, _ = result
        previous = self._phi(env)
        if new_seed:
            self.state['seed_step'] = 0
            self.seen_seeds.add(seed)
        self.state.update(env=env, directory=directory, seed=seed, episode=int(episode),
                          map_seed=int(map_seed), previous_phi=previous, episode_steps=0,
                          episode_removed=0., episode_pbrs=0.)
        return result

    def shape_reward(self, base, previous, following, *, terminated, truncated, gamma, config):
        if self.in_evaluation:
            raise RuntimeError('Evaluation attempted to invoke the training reward hook')
        if (self.state['env'] is None or config.enabled or gamma != .99 or
                previous is not None or following is not None):
            raise ValueError('Unexpected training reward callback context; B must remain disabled')
        current = self._phi(self.state['env'])
        reward, correction, detail = near_math.replace_near_reward(
            base, self.state['previous_phi'], current, terminated=terminated,
            truncated=truncated, gamma=gamma, mode=self.mode)
        # Preserve the original multiply-then-mean arithmetic; allow roundoff
        # rather than clipping away a possible one-ulp boundary residual.
        lower = -.8 if terminated else -.404
        if (not -1e-12 <= self.state['previous_phi'] <= .4 + 1e-12 or
                not -1e-12 <= current <= .4 + 1e-12 or
                correction < lower - 1e-12 or correction > 1e-12):
            raise ValueError('Near potential/correction outside analytic bounds')
        next_step = self.state['seed_step'] + 1
        row = dict(schema_version=1, kind='near_reward_transition', training_seed=self.state['seed'],
                   training_step=next_step, episode=self.state['episode'], map_seed=self.state['map_seed'],
                   **detail, replay_reward=float(reward), replay_done=bool(terminated))
        path = self.state['directory'] / 'near_reward_transitions.jsonl'
        # The old trainer owns each run directory with its run lock. Registry
        # serialization below has a different scope from this per-seed stream.
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            handle.flush()
        self.state.update(previous_phi=current, seed_step=next_step,
                          episode_steps=self.state['episode_steps'] + 1,
                          episode_removed=self.state['episode_removed'] + detail['removed_near'],
                          episode_pbrs=self.state['episode_pbrs'] + detail['potential_difference'])
        if next_step % self.log_interval == 0:
            text = (f'NEAR-REWARD mode={self.mode} seed={self.state["seed"]} step={next_step} '
                    f'original_base={float(base):.8f} removed_near={detail["removed_near"]:.8f} '
                    f'potential_difference={detail["potential_difference"]:.8f} '
                    f'correction={correction:.8f} replay_reward={float(reward):.8f} '
                    f'terminated={terminated} truncated={truncated}')
            if callable(getattr(self.trainer, 'message', None)):
                self.trainer.message(self.out, text)
            else:
                print(text, flush=True)
        return reward, correction

    def _evaluation_snapshot(self):
        state = self.state.copy()
        env = state['env']
        arrays = {key: np.array(getattr(env, key), copy=True) for key in
                  ('p', 'v', 'goal', 'prev_acc', 'captured', '_ever_in_ball')
                  if env is not None and hasattr(env, key)}
        scalars = {key: getattr(env, key) for key in ('t', 'seed')
                   if env is not None and hasattr(env, key)}
        return state, arrays, scalars, self.seen_seeds.copy()

    def _check_evaluation_snapshot(self, snapshot):
        state, arrays, scalars, seen = snapshot
        if self.state['env'] is not state['env'] or self.seen_seeds != seen:
            raise RuntimeError('Evaluation changed the training environment identity or seed state')
        for key, value in state.items():
            if key != 'env' and self.state[key] != value:
                raise RuntimeError(f'Evaluation changed training runtime field: {key}')
        env = state['env']
        for key, value in arrays.items():
            if not np.array_equal(np.asarray(getattr(env, key)), value):
                raise RuntimeError(f'Evaluation changed training environment array: {key}')
        for key, value in scalars.items():
            if getattr(env, key) != value:
                raise RuntimeError(f'Evaluation changed training environment scalar: {key}')

    def evaluate(self, *args, **kwargs):
        if self.in_evaluation:
            raise RuntimeError('Nested evaluation is unsupported')
        snapshot = self._evaluation_snapshot()
        self.in_evaluation = True
        completed = False
        try:
            result = self.original['evaluate'](*args, **kwargs)
            completed = True
            return result
        finally:
            try:
                self._check_evaluation_snapshot(snapshot)
                if completed:
                    self.original['append'](self.state['directory'] / 'evaluation_state_checks.jsonl',
                        dict(schema_version=1, kind='near_eval_state_check', passed=True,
                             training_seed=self.state['seed'], training_step=self.state['seed_step'],
                             mode=self.mode))
                    text = (f'NEAR-EVAL-STATE PASS seed={self.state["seed"]} '
                            f'step={self.state["seed_step"]} mode={self.mode}')
                    if callable(getattr(self.trainer, 'message', None)):
                        self.trainer.message(self.out, text)
                    else:
                        print(text, flush=True)
            finally:
                self.in_evaluation = False

    def append(self, path, obj):
        if obj.get('kind') == 'episode':
            if obj.get('seed') != self.state['seed'] or obj.get('step') != self.state['seed_step']:
                raise ValueError('Episode log does not match the active seed and callback step')
            obj = dict(obj, near_objective=dict(self.near),
                       near_removed_return=self.state['episode_removed'],
                       near_potential_difference_return=self.state['episode_pbrs'],
                       near_reward_steps_in_episode=self.state['episode_steps'],
                       shaping_return_semantics=SEMANTICS)
        return self.original['append'](path, obj)

    def write(self, path, obj):
        if Path(path).name == 'run_manifest.json':
            try:
                protocol_name = self.protocol.relative_to(ROOT).as_posix()
            except ValueError:
                protocol_name = str(self.protocol)
            obj = dict(obj, near_objective=dict(self.near),
                       preregistered_near_protocol=dict(path=protocol_name, sha256=self.protocol_sha256),
                       near_objective_runtime=dict(path='uav_planning/v4_near_study_runtime.py',
                                                   sha256=self.runtime_sha256,
                                                   adapter_path='uav_planning/v4_near_objective.py',
                                                   adapter_sha256=self.adapter_sha256),
                       evaluation_near_objective='original_PathC',
                       shaping_return_semantics=SEMANTICS,
                       near_reward_transition_log='maneuver_static_{seed}/near_reward_transitions.jsonl',
                       near_runtime_resume_supported=False,
                       registry_update_lock=dict(path=str(self.registry_lock_path),
                                                 maximum_wait_s=self.registry_wait_s))
        return self.original['write'](path, obj)

    def update_registry(self, *args, **kwargs):
        deadline = time.monotonic() + self.registry_wait_s
        while True:
            try:
                lock = RunLock(self.registry_lock_path,
                               metadata=f'near-study registry update pid={os.getpid()}')
                break
            except RuntimeError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'Near-study registry lock unavailable after '
                                       f'{self.registry_wait_s}s: {self.registry_lock_path}') from exc
                time.sleep(min(self.registry_poll_s, remaining))
        try:
            return self.original['update_registry'](*args, **kwargs)
        finally:
            lock.close()


def install_training_reward(trainer, cfg, out, protocol_path, *, registry_lock_path=None,
                            registry_wait_s=30.0, registry_poll_s=0.05):
    """Install explicit hooks once and return their runtime state/implementation.

    Both formal processes use the same default repository registry-lock path.
    ``registry_lock_path`` is exposed for isolated multiprocessing tests. The
    caller retains responsibility for the formal seed/budget/config freeze.
    """
    if getattr(trainer, '_near_study_runtime', None) is not None:
        raise ValueError('Near-study hooks were already installed on this trainer')
    runtime = NearStudyRuntime(trainer, cfg, out, protocol_path,
                               registry_lock_path=registry_lock_path,
                               registry_wait_s=registry_wait_s, registry_poll_s=registry_poll_s)
    for name in runtime.original:
        original = runtime.original[name]
        hook = getattr(runtime, name)

        @wraps(original)
        def installed(*args, _hook=hook, **kwargs):
            return _hook(*args, **kwargs)

        setattr(trainer, name, installed)
    trainer._near_study_runtime = runtime
    return runtime
