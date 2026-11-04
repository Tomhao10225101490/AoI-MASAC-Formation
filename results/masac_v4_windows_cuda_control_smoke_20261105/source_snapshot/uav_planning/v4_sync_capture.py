"""Opt-in, training-only potential shaping for frozen 4 m / 2 m/s capture.

Timeouts retain the next potential because the existing runner bootstraps them.
True terminations use the zero potential of an absorbing state. No environment,
observation, action, capture predicate, or evaluation reward is modified here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SyncCaptureConfig:
    enabled: bool = False
    coefficient: float = 0.5
    temperature: float = 0.1
    ball_m: float = 4.0
    speed_mps: float = 2.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('sync_capture.enabled must be a boolean')
        values = [self.coefficient, self.temperature, self.ball_m, self.speed_mps]
        if not np.isfinite(values).all() or self.coefficient < 0 or min(values[1:]) <= 0:
            raise ValueError('sync_capture requires finite nonnegative coefficient and positive scales')
        if self.ball_m != 4.0 or self.speed_mps != 2.0:
            raise ValueError('sync_capture scales must match frozen capture 4m/2mps')

    def to_dict(self):
        return asdict(self)


def sync_from_protocol(value=None):
    return SyncCaptureConfig(**(value or {}))


def capture_potential(positions, goals, velocities, config: SyncCaptureConfig):
    """Bounded smooth worst-agent normalized distance/speed potential in (0, 1]."""
    positions, goals, velocities = (np.asarray(x, dtype=np.float64) for x in (positions, goals, velocities))
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions):
        raise ValueError('expected nonempty (n_agents, 3) positions')
    if goals.shape != positions.shape or velocities.shape != positions.shape:
        raise ValueError('position/goal/velocity shapes must match')
    if not all(np.isfinite(x).all() for x in (positions, goals, velocities)):
        raise ValueError('nonfinite capture-potential state')
    # sqrt(||x||^2 + eps^2) - eps is smooth even at x=0, and is exactly zero there.
    eps = 1e-8
    norm = lambda x: np.sqrt(np.sum(x * x, axis=-1) + eps * eps) - eps
    q = np.concatenate((norm(goals - positions) / config.ball_m, norm(velocities) / config.speed_mps))
    largest = float(np.max(q))
    burden = largest + config.temperature * np.log(np.mean(np.exp((q - largest) / config.temperature)))
    return float(1.0 / (1.0 + max(0.0, burden)))


def shape_reward(reward, previous, following, *, terminated, truncated, gamma, config):
    """Return (training reward, shaping); default-off preserves the reward object.

    `truncated` is explicit to prevent accidental conflation with termination.
    The runner's replay mask is `terminated`, so a timeout keeps its bootstrap.
    """
    if not config.enabled:
        return reward, 0.0
    if not 0 <= gamma < 1 or not np.isfinite([reward, previous, following]).all():
        raise ValueError('invalid shaping reward/potential/discount')
    next_potential = 0.0 if terminated else following
    shaping = config.coefficient * (gamma * next_potential - previous)
    result = reward + shaping
    if not np.isfinite(result):
        raise ValueError('nonfinite shaped reward')
    return result, float(shaping)


def occupancy(env):
    distance = np.linalg.norm(env.goal - env.p, axis=-1)
    speed = np.linalg.norm(env.v, axis=-1)
    return int(np.count_nonzero((distance < 4.0) & (speed < 2.0)))


def failure_reason(record):
    if record.get('success'):
        return 'success'
    for name in ('uav_collision', 'static_collision', 'dynamic_collision', 'boundary_collision'):
        if record.get(name):
            return name
    if record.get('timeout'):
        ever = record.get('ever_in_ball_by_uav') or []
        if ever and all(ever):
            return 'timeout_after_ball_all'
        return 'timeout_after_ball_any_only' if any(ever) else 'timeout_before_ball'
    return 'non_success_other'
