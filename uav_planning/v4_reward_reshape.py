"""Path-C capture/reward/horizon reshape for SharedMASAC learnability floor.

Does NOT edit frozen uav_planning/v4_environment.py (L0 continue source hashes).
Subclass of V4Mission with auditable training-time shaping so empty-map
approach + capture can learn before returning to the frozen L0 gate.

Frozen baseline (v4_environment.V4Mission):
  progress      = (before - after) / 2
  living        = -0.02
  acc_penalty   = -0.002 * ||acc||^2
  capture_delta = +5 * (arrived - prev)   # can be -5 on leave
  collision     = -20
  team_success  = +20
  arrived       = (dist < 4) AND (speed < 2), not sticky, simultaneous

C reshape defaults (this module; experiment-local):
  progress_scale     = 1.0          # was 0.5
  living_cost        = 0.005        # was 0.02
  near_goal_radius_m = 15.0
  near_goal_bonus    = 0.4          # dense potential inside 15 m
  ball_enter_bonus   = 3.0          # rising edge: dist < train_ball (any speed)
  capture_bonus      = 8.0          # rising edge: soft capture (latched)
  team_success_bonus = 50.0         # was 20
  train_ball_m       = 6.0          # was 4; TRAINING+this-exp eval
  train_speed_mps    = 3.5          # was 2
  capture_latch      = True         # sticky; no -5 on leave
  horizon            = 350          # unchanged (40 m goals; travel budget ample)

Protocols may set train_ball_m / train_speed_mps / capture_latch via ReshapeConfig.
Empty-floor Path C used soft 6/3.5+latch. L0-return Path C keeps reward reshape but
uses frozen capture 4/2 with L0-default latch=False for train+gate.
Standard observation only; no Attention / maneuverability / L1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from uav_planning.environment import swept_distance
from uav_planning.structured import box_distance, limited, segment_box_hit
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_sanity_maps import inject_single_obstacle, sanity_config


# Frozen L0 gate (documentation / future return only — not used by this env).
FROZEN_GOAL_BALL_M = 4.0
FROZEN_CAPTURE_SPEED_MPS = 2.0


@dataclass(frozen=True)
class ReshapeConfig:
    """Auditable numeric reshape for path C."""

    progress_scale: float = 1.0
    living_cost: float = 0.005
    acc_penalty: float = 0.002
    near_goal_radius_m: float = 15.0
    near_goal_bonus: float = 0.4
    ball_enter_bonus: float = 3.0
    capture_bonus: float = 8.0
    team_success_bonus: float = 50.0
    collision_penalty: float = 20.0
    train_ball_m: float = 6.0
    train_speed_mps: float = 3.5
    capture_latch: bool = True
    horizon: int = 350

    def to_dict(self):
        return asdict(self)


DEFAULT_RESHAPE = ReshapeConfig()


def soft_capture_mask(positions, goals, velocities, ball_m, speed_mps):
    after = np.linalg.norm(np.asarray(goals) - np.asarray(positions), axis=-1)
    speed = np.linalg.norm(np.asarray(velocities), axis=-1)
    return (after < ball_m) & (speed < speed_mps)


def in_ball_mask(positions, goals, ball_m):
    after = np.linalg.norm(np.asarray(goals) - np.asarray(positions), axis=-1)
    return after < ball_m


class V4MissionRewardReshape(V4Mission):
    """V4Mission with path-C reward / soft-capture / latch reshape."""

    def __init__(self, config=None, reshape: ReshapeConfig | None = None):
        super().__init__(config)
        self.reshape = reshape or DEFAULT_RESHAPE
        # Allow protocol to override horizon without touching SceneConfig frozen defaults.
        if self.reshape.horizon is not None:
            self.horizon = int(self.reshape.horizon)
        self._ever_in_ball = np.zeros(self.n, dtype=bool)

    def reset(self, seed):
        obs = super().reset(seed)
        if self.reshape.horizon is not None:
            self.horizon = int(self.reshape.horizon)
        self._ever_in_ball = np.zeros(self.n, dtype=bool)
        self.captured = np.zeros(self.n, dtype=bool)
        return obs

    def step(self, action):
        rs = self.reshape
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n, 3) or not np.isfinite(action).all():
            raise ValueError('Expected finite per-UAV 3D actions')
        old = self.p.copy()
        old_dynamic = self.dynamic.copy()
        before = np.linalg.norm(self.goal - self.p, axis=-1)
        old_v = self.v.copy()
        prev_captured = self.captured.copy()
        prev_in_ball = self._ever_in_ball.copy()
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
        self.minimum_clearance = min(
            self.minimum_clearance, min_boundary, min_static, min_dynamic, minsep - 2 * self.radius
        )

        after = np.linalg.norm(self.goal - self.p, axis=-1)
        in_ball_now = in_ball_mask(self.p, self.goal, rs.train_ball_m)
        # Rising edge of first ball entry (distance-only), sticky tracker for bonus once.
        first_ball_entry = in_ball_now & ~prev_in_ball
        self._ever_in_ball = prev_in_ball | in_ball_now

        arrived_now = soft_capture_mask(
            self.p, self.goal, self.v, rs.train_ball_m, rs.train_speed_mps
        )
        if rs.capture_latch:
            self.captured = prev_captured | arrived_now
            capture_delta = self.captured.astype(float) - prev_captured.astype(float)
            arrived_for_success = self.captured
        else:
            self.captured = arrived_now
            capture_delta = arrived_now.astype(float) - prev_captured.astype(float)
            arrived_for_success = arrived_now

        progress = rs.progress_scale * (before - after)
        living = -rs.living_cost
        acc_pen = -rs.acc_penalty * np.sum(acc * acc, axis=-1)
        # Dense near-goal potential (continuous; stronger close in).
        if rs.near_goal_radius_m > 0 and rs.near_goal_bonus != 0:
            near = rs.near_goal_bonus * np.clip(1.0 - after / rs.near_goal_radius_m, 0.0, 1.0)
        else:
            near = 0.0
        ball_bonus = rs.ball_enter_bonus * first_ball_entry.astype(float)
        cap_bonus = rs.capture_bonus * capture_delta
        reward = progress + living + acc_pen + near + ball_bonus + cap_bonus

        collision = peer or static or moving or boundary
        if collision:
            reward -= rs.collision_penalty
        success = bool(arrived_for_success.all() and not collision)
        if success:
            reward += rs.team_success_bonus

        self.length_by_uav += np.linalg.norm(self.p - old, axis=-1)
        self.length = float(self.length_by_uav.sum())
        self.jerk += float(np.sum(((acc - self.prev_acc) / self.dt) ** 2) * self.dt)
        self.prev_acc = acc
        self.t += 1
        info = dict(
            success=success,
            uav_collision=peer,
            static_collision=static,
            dynamic_collision=moving,
            boundary_collision=boundary,
            arrived_fraction=float(arrived_for_success.mean()),
            arrived_by_uav=arrived_for_success.tolist(),
            arrived_now_by_uav=arrived_now.tolist(),
            in_ball_now_by_uav=in_ball_now.tolist(),
            ever_in_ball_by_uav=self._ever_in_ball.tolist(),
            capture_latch=bool(rs.capture_latch),
            reshape=rs.to_dict(),
            train_ball_m=float(rs.train_ball_m),
            train_speed_mps=float(rs.train_speed_mps),
            frozen_eval_note=(
                f'This experiment eval uses soft train thresholds '
                f'(ball<{rs.train_ball_m}, speed<{rs.train_speed_mps}, latch={rs.capture_latch}). '
                f'Frozen L0 gate remains dist<{FROZEN_GOAL_BALL_M} AND speed<{FROZEN_CAPTURE_SPEED_MPS}.'
            ),
            path_length=self.length,
            path_length_by_uav=self.length_by_uav.tolist(),
            time=self.t * self.dt,
            min_separation=self.minimum_separation,
            min_clearance=self.minimum_clearance,
            jerk_integral=self.jerk,
            remaining_distance=float(after.mean()),
            scene_sha256=self.scene.scene_hash,
        )
        return (
            self.observe(),
            float(np.mean(reward)),
            bool(collision or success),
            self.t >= self.horizon,
            info,
        )


def reshape_from_protocol(protocol_reshape: dict | None) -> ReshapeConfig:
    base = DEFAULT_RESHAPE.to_dict()
    if protocol_reshape:
        for key, value in protocol_reshape.items():
            if key not in base:
                raise ValueError(f'Unknown reshape key {key!r}')
            base[key] = value
    return ReshapeConfig(**base)


def make_reshape_env(kind, seed, sensor_range=400., reshape: ReshapeConfig | None = None, **overrides):
    """Build empty / single-obstacle env with path-C reshape."""
    rs = reshape or DEFAULT_RESHAPE
    cfg = sanity_config(kind, sensor_range=sensor_range, horizon=rs.horizon, **overrides)
    env = V4MissionRewardReshape(cfg, reshape=rs)
    obs = env.reset(int(seed))
    if kind == 'single_obstacle':
        inject_single_obstacle(env, int(seed))
        obs = env.observe()
    return env, obs



def make_reshape_l0_env(family, seed, sensor_range=400., reshape: ReshapeConfig | None = None, **overrides):
    """Build a frozen L0 family scene with path-C reward reshape (capture knobs from reshape)."""
    from uav_planning.v4_maps import FAMILIES, default_config
    if family not in FAMILIES:
        raise ValueError(f'Unknown L0 family {family!r}; expected one of {FAMILIES}')
    rs = reshape or DEFAULT_RESHAPE
    cfg = default_config(family, 'L0', sensor_range=sensor_range, horizon=rs.horizon, **overrides)
    env = V4MissionRewardReshape(cfg, reshape=rs)
    obs = env.reset(int(seed))
    return env, obs


__all__ = [
    'FROZEN_GOAL_BALL_M',
    'FROZEN_CAPTURE_SPEED_MPS',
    'ReshapeConfig',
    'DEFAULT_RESHAPE',
    'soft_capture_mask',
    'in_ball_mask',
    'V4MissionRewardReshape',
    'reshape_from_protocol',
    'make_reshape_env',
    'make_reshape_l0_env',
]
