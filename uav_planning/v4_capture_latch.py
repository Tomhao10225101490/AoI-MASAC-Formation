"""Capture-latch isolation for V4Mission without editing frozen v4_environment.py.

Baseline (capture_latch=False) matches frozen semantics exactly:
  arrived = (dist < 4) AND (speed < 2)   # simultaneous per-step
  delta   = arrived - previous_captured  # can be -1 → -5 reward on leave
  captured = arrived                     # not sticky

Latch (capture_latch=True):
  once a UAV satisfies the ball+speed gate, captured stays True forever
  (within the episode). Leaving the ball or speeding up does NOT emit -5.
  Team success still requires all agents captured (latched) and no collision.

Standard observation only; no Attention / maneuverability features.
"""
from __future__ import annotations

import numpy as np

from uav_planning.environment import swept_distance
from uav_planning.structured import box_distance, limited, segment_box_hit
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import default_config


GOAL_BALL_M = 4.0
CAPTURE_SPEED_MPS = 2.0


def capture_mask(positions, goals, velocities, ball_m=GOAL_BALL_M, speed_mps=CAPTURE_SPEED_MPS):
    after = np.linalg.norm(np.asarray(goals) - np.asarray(positions), axis=-1)
    speed = np.linalg.norm(np.asarray(velocities), axis=-1)
    return (after < ball_m) & (speed < speed_mps)


class V4MissionCaptureVariant(V4Mission):
    """V4Mission with optional sticky capture latch.

    When capture_latch is False, step() reproduces frozen capture/reward algebra.
    When True, captured is OR-accumulated; delta is never negative from leave.
    """

    def __init__(self, config=None, capture_latch=False):
        super().__init__(config)
        self.capture_latch = bool(capture_latch)

    def reset(self, seed):
        obs = super().reset(seed)
        self.capture_latch = bool(self.capture_latch)
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n, 3) or not np.isfinite(action).all():
            raise ValueError('Expected finite per-UAV 3D actions')
        old = self.p.copy()
        old_dynamic = self.dynamic.copy()
        before = np.linalg.norm(self.goal - self.p, axis=-1)
        old_v = self.v.copy()
        prev_captured = self.captured.copy()
        self.v = limited(self.v, np.clip(action, -1, 1), self.dt, self.amax, self.vmax)
        acc = (self.v - old_v) / self.dt
        self.p += self.dt * self.v
        self._advance_dynamic()

        sep = swept_distance(old[:, None] - old[None], self.p[:, None] - self.p[None])
        np.fill_diagonal(sep, np.inf)
        minsep = float(sep.min())
        self.minimum_separation = min(self.minimum_separation, minsep)
        peer = bool(np.any(sep < 2 * self.radius + self.margin))
        inflation = self.radius + self.margin
        static = bool(len(self.boxes) and segment_box_hit(
            old, self.p, self.lo - inflation, self.hi + inflation).any())
        if len(self.dynamic):
            moving_distance = swept_distance(
                old[:, None] - old_dynamic[None], self.p[:, None] - self.dynamic[None])
            moving = bool(np.any(moving_distance < self.dr[None, :] + inflation))
            min_dynamic = float(np.min(moving_distance - self.dr[None, :] - self.radius))
        else:
            moving = False
            min_dynamic = float('inf')
        boundary = bool(np.any(self.p < inflation) or np.any(self.p > self.world - inflation))
        min_boundary = float(np.minimum(self.p - self.radius, self.world - self.radius - self.p).min())
        min_static = (float(np.min(box_distance(self.p, self.lo, self.hi)) - self.radius)
                      if len(self.boxes) else float('inf'))
        self.minimum_clearance = min(self.minimum_clearance, min_boundary, min_static,
                                     min_dynamic, minsep - 2 * self.radius)
        after = np.linalg.norm(self.goal - self.p, axis=-1)
        arrived_now = capture_mask(self.p, self.goal, self.v)
        if self.capture_latch:
            self.captured = prev_captured | arrived_now
            # Latch: reward only the rising edge; never punish leave.
            delta = self.captured.astype(float) - prev_captured.astype(float)
            arrived_for_success = self.captured
        else:
            self.captured = arrived_now
            delta = arrived_now.astype(float) - prev_captured.astype(float)
            arrived_for_success = arrived_now
        reward = (before - after) / 2 - .02 - .002 * np.sum(acc * acc, axis=-1) + 5 * delta
        collision = peer or static or moving or boundary
        if collision:
            reward -= 20
        success = bool(arrived_for_success.all() and not collision)
        if success:
            reward += 20
        self.length_by_uav += np.linalg.norm(self.p - old, axis=-1)
        self.length = float(self.length_by_uav.sum())
        self.jerk += float(np.sum(((acc - self.prev_acc) / self.dt) ** 2) * self.dt)
        self.prev_acc = acc
        self.t += 1
        info = dict(
            success=success, uav_collision=peer, static_collision=static,
            dynamic_collision=moving, boundary_collision=boundary,
            arrived_fraction=float(arrived_for_success.mean()),
            arrived_by_uav=arrived_for_success.tolist(),
            arrived_now_by_uav=arrived_now.tolist(),
            capture_latch=bool(self.capture_latch),
            path_length=self.length, path_length_by_uav=self.length_by_uav.tolist(),
            time=self.t * self.dt, min_separation=self.minimum_separation,
            min_clearance=self.minimum_clearance, jerk_integral=self.jerk,
            remaining_distance=float(after.mean()), scene_sha256=self.scene.scene_hash,
        )
        return (
            self.observe(),
            float(np.mean(reward)),
            bool(collision or success),
            self.t >= self.horizon,
            info,
        )


def make_standard_env(family, sensor_range=400., capture_latch=False):
    cfg = default_config(family, 'L0', sensor_range=sensor_range)
    return V4MissionCaptureVariant(cfg, capture_latch=capture_latch)


__all__ = [
    'GOAL_BALL_M',
    'CAPTURE_SPEED_MPS',
    'capture_mask',
    'V4MissionCaptureVariant',
    'make_standard_env',
]
