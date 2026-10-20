"""Cluttered leader-follower transit with a geometry-dependent command link.

Gym-style API, no Gymnasium dependency:

    obs, info = env.reset(seed=None, options=None)
    obs, rewards, terminated, truncated, info = env.step(actions)

What the episode is
-------------------
A leader flies a sequence of waypoints across a 1200 m by 800 m field of
static buildings and crossing traffic. Followers keep a wedge behind the
leader. Wind is drawn once per episode. This is the kind of static-plus-dynamic
obstacle field used by recent formation RL, with two additions those papers
usually leave out:

* The leader state reaches each follower on a downlink whose success
  probability falls with distance and drops again when a building intersects
  the segment (non-line-of-sight). A lost packet keeps the previous leader
  state and increments that follower's Age of Information.
* The actor is given that age. The reward still uses the true leader pose,
  so a stale slot and the true slot disagree. That disagreement is the task.

Onboard sensing (obstacles, nearest teammate, a noisy wind estimate) is
limited by ``sensor_range`` and is not sent on the lossy link.

Actions are outer-loop commands, which is how most 2024–2026 UAV RL papers
treat an assumed inner autopilot: normalized longitudinal acceleration and
yaw rate, integrated at ``dt``. They are not instantaneous jumps in speed
and heading.

``terminated`` is success or collision. ``truncated`` is the step limit.
Success means every waypoint has been captured, every follower is inside
``formation_tol`` of the slot defined by the *true* leader pose, and the
episode never collided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from envs.channel import packet_success_probability
from envs.geometry import (
    desired_slot,
    make_formation_offsets,
    segment_hits_disk,
    wrap_angle,
)
from envs.scenario import default_obstacles, default_waypoints


class ObsIndex:
    """Column indices of one local observation. Same length for every agent."""

    X = 0
    Y = 1
    V = 2
    SIN_H = 3
    COS_H = 4
    REF_REL_X = 5
    REF_REL_Y = 6
    REF_V = 7
    REF_SIN = 8
    REF_COS = 9
    AOI = 10
    SLOT_X = 11
    SLOT_Y = 12
    DES_REL_X = 13
    DES_REL_Y = 14
    WP_FRAC = 15
    WIND_X = 16
    WIND_Y = 17
    TEAMMATE_REL_X = 18
    TEAMMATE_REL_Y = 19
    TEAMMATE_DIST = 20
    TEAMMATE_PRESENT = 21
    IS_LEADER = 22
    OBS_START = 23
    # rel_x, rel_y, vx, vy, radius, present — only if inside sensor range
    OBS_FEATS = 6

    @staticmethod
    def dim(n_nearest_obstacles: int) -> int:
        return ObsIndex.OBS_START + ObsIndex.OBS_FEATS * int(n_nearest_obstacles)


def global_state_dim(
    n_agents: int, n_followers: int, max_obstacles: int, max_waypoints: int
) -> int:
    """Privileged critic input length. Fixed once the config is fixed."""
    return (
        5 * n_agents
        + 2
        + 1
        + 2 * max_waypoints
        + 5 * max_obstacles
        + 7 * n_followers
    )


@dataclass
class EnvConfig:
    """Environment knobs. Unknown YAML keys are rejected by the loader."""

    map_width: float = 1200.0
    map_height: float = 800.0
    n_followers: int = 4
    dt: float = 0.2
    max_steps: int = 500
    v_min: float = 0.0
    v_max: float = 16.0
    max_accel: float = 6.0
    max_yaw_rate: float = 1.2
    uav_radius: float = 4.0
    wp_radius: float = 30.0
    formation_tol: float = 18.0
    formation_spacing: float = 32.0
    formation_scale: float = 96.0
    n_nearest_obstacles: int = 3
    max_obstacles: int = 11
    max_waypoints: int = 4
    obstacles: list = field(default_factory=default_obstacles)
    extra_obstacles: list = field(default_factory=list)
    formation_offsets: list | None = None
    waypoints: list = field(default_factory=default_waypoints)
    channel_mode: str = "distance"
    p0: float = 0.95
    d0: float = 200.0
    nlos_factor: float = 0.2
    blockage_margin: float = 0.0
    use_blockage: bool = True
    randomize_channel: bool = True
    p0_low: float = 0.45
    p0_high: float = 1.0
    aoi_cap: int = 40
    obs_mode: str = "aoi"
    use_aoi: bool = True
    sensor_range: float = 250.0
    start_xy: tuple = (150.0, 420.0)
    start_speed: float = 10.0
    start_noise: float = 12.0
    wind_speed: float = 0.0
    wind_heading: float = 0.0
    wind_max: float = 4.0
    randomize_wind: bool = True
    wind_obs_noise: float = 0.5
    seed: int = 0
    # Initial weights for the new dynamics. Not a tuned paper result.
    w_progress: float = 0.25
    w_heading: float = 0.05
    w_formation: float = 0.02
    w_form_progress: float = 0.25
    w_smooth: float = 0.02
    w_step: float = 0.01
    w_clearance: float = 0.5
    clearance_margin: float = 20.0
    formation_bonus: float = 0.15
    collision_penalty: float = 30.0
    success_bonus: float = 40.0
    wp_bonus: float = 8.0
    oob_penalty: float = 1.0
    obstacle_speed_norm: float = 10.0

    def __post_init__(self) -> None:
        self.start_xy = (float(self.start_xy[0]), float(self.start_xy[1]))
        self.waypoints = [(float(p[0]), float(p[1])) for p in self.waypoints]
        if self.n_followers < 1:
            raise ValueError("n_followers must be >= 1")
        if self.n_nearest_obstacles < 1:
            raise ValueError("n_nearest_obstacles must be >= 1")
        if self.max_obstacles < 0:
            raise ValueError("max_obstacles must be >= 0")
        if len(self.waypoints) < 1:
            raise ValueError("at least one waypoint is required")
        if len(self.waypoints) > self.max_waypoints:
            raise ValueError(
                f"{len(self.waypoints)} waypoints but max_waypoints={self.max_waypoints}. "
                "Raising max_waypoints changes the critic input size."
            )
        n_obs = len(self.obstacles) + len(self.extra_obstacles)
        if n_obs > self.max_obstacles:
            raise ValueError(
                f"{n_obs} obstacles (including extra_obstacles) but "
                f"max_obstacles={self.max_obstacles}."
            )
        if self.obs_mode not in ("aoi", "perfect", "hold", "extrapolate"):
            raise ValueError(f"bad obs_mode {self.obs_mode!r}")
        if self.channel_mode not in ("distance", "independent", "perfect"):
            raise ValueError(f"bad channel_mode {self.channel_mode!r}")
        if self.aoi_cap < 1:
            raise ValueError("aoi_cap must be >= 1")
        if self.formation_offsets is not None and len(self.formation_offsets) != self.n_followers:
            raise ValueError("formation_offsets length must equal n_followers")
        if self.max_steps < 1 or self.dt <= 0.0:
            raise ValueError("max_steps must be >= 1 and dt must be positive")
        if self.v_max < self.v_min:
            raise ValueError("v_max must be >= v_min")
        if self.map_width <= 0.0 or self.map_height <= 0.0:
            raise ValueError("map size must be positive")
        if self.sensor_range <= 0.0:
            raise ValueError("sensor_range must be positive")
        if self.wind_max <= 0.0:
            raise ValueError("wind_max must be positive")


def env_config_from_dict(data: dict) -> EnvConfig:
    known = {f.name for f in fields(EnvConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise KeyError(f"Unknown env config keys: {unknown}")
    return EnvConfig(**data)


@dataclass
class _UAV:
    x: float
    y: float
    v: float
    heading: float


@dataclass
class _Obstacle:
    x: float
    y: float
    radius: float
    vx: float
    vy: float

    @classmethod
    def from_dict(cls, raw: dict) -> "_Obstacle":
        return cls(
            x=float(raw["x"]),
            y=float(raw["y"]),
            radius=float(raw["radius"]),
            vx=float(raw.get("vx", 0.0)),
            vy=float(raw.get("vy", 0.0)),
        )


@dataclass
class _Link:
    last_x: float
    last_y: float
    last_v: float
    last_heading: float
    aoi: int
    blocked: int = 0


class UAVFormationEnv:
    """Waypoint transit, wedge formation, blockage-aware lossy leader link."""

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config if config is not None else EnvConfig()
        self.n_followers = int(self.cfg.n_followers)
        self.n_agents = self.n_followers + 1
        self.action_dim = 2
        self.obs_dim = ObsIndex.dim(self.cfg.n_nearest_obstacles)
        self.obs_dims = [self.obs_dim] * self.n_agents
        self.global_dim = global_state_dim(
            self.n_agents,
            self.n_followers,
            int(self.cfg.max_obstacles),
            int(self.cfg.max_waypoints),
        )
        if self.cfg.formation_offsets is None:
            self.offsets = make_formation_offsets(
                self.n_followers, spacing=float(self.cfg.formation_spacing)
            )
        else:
            self.offsets = [(float(p[0]), float(p[1])) for p in self.cfg.formation_offsets]
        raw_obstacles = list(self.cfg.obstacles) + list(self.cfg.extra_obstacles)
        self._obstacle_init = [_Obstacle.from_dict(o) for o in raw_obstacles]
        self.waypoints = [(float(p[0]), float(p[1])) for p in self.cfg.waypoints]
        self.diag = float(math.hypot(self.cfg.map_width, self.cfg.map_height))
        self._rng = np.random.default_rng(self.cfg.seed)
        self.uavs: list[_UAV] = []
        self.obstacles: list[_Obstacle] = []
        self.links: list[_Link] = []
        self._wp_index = 0
        self._wp_done = 0
        self._step_count = 0
        self.wind_x = 0.0
        self.wind_y = 0.0
        self._prev_wp_dist = 0.0
        self._prev_form_err = np.zeros(self.n_followers, dtype=np.float64)
        self.ep_p0 = float(self.cfg.p0)
        self.ep_channel_mode = self.cfg.channel_mode
        self.ep_obs_mode = self.cfg.obs_mode
        self.ep_use_aoi = bool(self.cfg.use_aoi)
        self.ep_use_blockage = bool(self.cfg.use_blockage)
        self._reset_stat_counters()

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[list[np.ndarray], dict]:
        """Reset. ``options`` may override the link for this episode only.

        Supported option keys: ``p0``, ``channel_mode``, ``obs_mode``,
        ``use_aoi``, ``use_blockage``, ``randomize_channel``, ``d0``,
        ``p0_low``, ``p0_high``. Passing ``seed`` rebuilds the RNG.
        """
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._apply_options(options or {})
        self._sample_wind()
        self._spawn()
        self._wp_index = 0
        self._wp_done = 0
        self._deliver_initial_leader_state()
        self._step_count = 0
        self._reset_stat_counters()
        self._prev_wp_dist = self._waypoint_distance()
        self._prev_form_err = self._formation_errors()
        return self._observations(), self._info_skeleton()

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, bool, bool, dict]:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n_agents, self.action_dim):
            raise ValueError(
                f"actions shape {actions.shape} != {(self.n_agents, self.action_dim)}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("actions contain NaN or Inf")
        actions = np.clip(actions, -1.0, 1.0)

        self._integrate(actions)
        self._move_obstacles()
        oob = self._apply_bounds()
        hit_obstacle, hit_uav = self._collision_flags()
        captured = self._advance_waypoints()
        packet_ok = self._update_channel()
        errors = self._formation_errors()
        wp_dist = self._waypoint_distance()
        rewards = self._rewards(
            actions=actions,
            errors=errors,
            wp_dist=wp_dist,
            captured=captured,
            hit_obstacle=hit_obstacle,
            hit_uav=hit_uav,
            oob=oob,
        )

        self._step_count += 1
        collided = bool(hit_obstacle or hit_uav)
        in_formation = bool(np.all(errors <= self.cfg.formation_tol))
        route_done = self._wp_done >= len(self.waypoints)
        success = bool(route_done and in_formation and not collided)
        timeout = bool(
            self._step_count >= self.cfg.max_steps and not success and not collided
        )
        terminated = bool(success or collided)
        truncated = bool(timeout)
        self._accumulate(errors, in_formation)
        self._prev_wp_dist = wp_dist
        self._prev_form_err = errors

        info = self._info_skeleton()
        info["collision"] = collided
        info["success"] = success
        info["packet_success"] = packet_ok.copy()
        info["formation_errors"] = errors.copy()
        info["waypoints_done"] = int(self._wp_done)
        if terminated or truncated:
            info["episode"] = self._episode_dict(
                success, collided, timeout, hit_obstacle, hit_uav
            )
        return self._observations(), rewards, terminated, truncated, info

    def global_state(self) -> np.ndarray:
        """Privileged state for the centralized critic. Actors never see this."""
        cfg = self.cfg
        parts: list[float] = []
        for u in self.uavs:
            parts.extend(
                [
                    u.x / cfg.map_width,
                    u.y / cfg.map_height,
                    u.v / cfg.v_max,
                    math.sin(u.heading),
                    math.cos(u.heading),
                ]
            )
        parts.extend([self.wind_x / cfg.wind_max, self.wind_y / cfg.wind_max])
        parts.append(self._wp_done / len(self.waypoints))
        for k in range(cfg.max_waypoints):
            if k < len(self.waypoints):
                parts.extend(
                    [self.waypoints[k][0] / cfg.map_width, self.waypoints[k][1] / cfg.map_height]
                )
            else:
                parts.extend([0.0, 0.0])
        for k in range(cfg.max_obstacles):
            if k < len(self.obstacles):
                o = self.obstacles[k]
                parts.extend(
                    [
                        o.x / cfg.map_width,
                        o.y / cfg.map_height,
                        o.vx / cfg.obstacle_speed_norm,
                        o.vy / cfg.obstacle_speed_norm,
                        o.radius / self.diag,
                    ]
                )
            else:
                parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        for link in self.links:
            parts.extend(
                [
                    link.aoi / cfg.aoi_cap,
                    link.last_x / cfg.map_width,
                    link.last_y / cfg.map_height,
                    link.last_v / cfg.v_max,
                    math.sin(link.last_heading),
                    math.cos(link.last_heading),
                    float(link.blocked),
                ]
            )
        state = np.asarray(parts, dtype=np.float32)
        if state.shape != (self.global_dim,):
            raise RuntimeError(f"global state length {state.shape[0]} != {self.global_dim}")
        return state

    def reference_leader_state(self, follower_index: int) -> np.ndarray:
        """Leader (x, y, v, heading) written into that follower's observation."""
        return self._reference_pose(follower_index)

    def link_blocked(self, follower_index: int) -> bool:
        """True when a building disk intersects the follower–leader segment."""
        leader = self.uavs[0]
        follower = self.uavs[follower_index + 1]
        return self._segment_blocked(leader.x, leader.y, follower.x, follower.y)

    def snapshot(self) -> dict:
        wp = self.waypoints[min(self._wp_index, len(self.waypoints) - 1)]
        return {
            "uav_xy": np.array([[u.x, u.y] for u in self.uavs], dtype=np.float32),
            "heading": np.array([u.heading for u in self.uavs], dtype=np.float32),
            "obstacle_xy": np.array(
                [[o.x, o.y] for o in self.obstacles], dtype=np.float32
            ).reshape(-1, 2),
            "obstacle_r": np.array([o.radius for o in self.obstacles], dtype=np.float32),
            "aoi": np.array([link.aoi for link in self.links], dtype=np.float32),
            "goal": np.array(wp, dtype=np.float32),
            "waypoints": np.array(self.waypoints, dtype=np.float32),
            "slots": self._true_slots().astype(np.float32),
        }

    # ------------------------------------------------------------------
    def _apply_options(self, options: dict) -> None:
        cfg = self.cfg
        mode = str(options.get("channel_mode", cfg.channel_mode))
        obs_mode = str(options.get("obs_mode", cfg.obs_mode))
        if mode not in ("distance", "independent", "perfect"):
            raise ValueError(f"bad channel_mode {mode!r}")
        if obs_mode not in ("aoi", "perfect", "hold", "extrapolate"):
            raise ValueError(f"bad obs_mode {obs_mode!r}")
        randomize = bool(options.get("randomize_channel", cfg.randomize_channel))
        if "p0" in options and options["p0"] is not None:
            p0 = float(options["p0"])
        elif randomize and mode != "perfect":
            low = float(options.get("p0_low", cfg.p0_low))
            high = float(options.get("p0_high", cfg.p0_high))
            if high < low:
                raise ValueError("p0_high must be >= p0_low")
            p0 = float(self._rng.uniform(low, high))
        else:
            p0 = float(cfg.p0)
        if not 0.0 <= p0 <= 1.0:
            raise ValueError(f"p0 must be in [0, 1], got {p0}")
        self.ep_channel_mode = mode
        self.ep_p0 = p0
        self.ep_d0 = float(options.get("d0", cfg.d0))
        self.ep_obs_mode = obs_mode
        if "use_aoi" in options and options["use_aoi"] is not None:
            self.ep_use_aoi = bool(options["use_aoi"])
        else:
            self.ep_use_aoi = bool(cfg.use_aoi)
        if "use_blockage" in options and options["use_blockage"] is not None:
            self.ep_use_blockage = bool(options["use_blockage"])
        else:
            self.ep_use_blockage = bool(cfg.use_blockage)

    def _sample_wind(self) -> None:
        cfg = self.cfg
        if cfg.randomize_wind:
            speed = float(self._rng.uniform(0.0, cfg.wind_max))
            heading = float(self._rng.uniform(-math.pi, math.pi))
        else:
            speed = float(cfg.wind_speed)
            heading = float(cfg.wind_heading)
        self.wind_x = speed * math.cos(heading)
        self.wind_y = speed * math.sin(heading)

    def _spawn(self) -> None:
        cfg = self.cfg
        leader = self._sample_leader_pose()
        self.uavs = [leader]
        for offset in self.offsets:
            slot = desired_slot((leader.x, leader.y), leader.heading, offset)
            placed = False
            for _ in range(40):
                noise = self._rng.uniform(-cfg.start_noise, cfg.start_noise, size=2)
                x = float(slot[0] + noise[0])
                y = float(slot[1] + noise[1])
                heading = wrap_angle(leader.heading + float(self._rng.uniform(-0.05, 0.05)))
                if self._point_clear(x, y):
                    self.uavs.append(_UAV(x, y, cfg.start_speed, heading))
                    placed = True
                    break
            if not placed:
                x = float(np.clip(slot[0], cfg.uav_radius, cfg.map_width - cfg.uav_radius))
                y = float(np.clip(slot[1], cfg.uav_radius, cfg.map_height - cfg.uav_radius))
                self.uavs.append(_UAV(x, y, cfg.start_speed, leader.heading))
        self.obstacles = [
            _Obstacle(o.x, o.y, o.radius, o.vx, o.vy) for o in self._obstacle_init
        ]

    def _sample_leader_pose(self) -> _UAV:
        cfg = self.cfg
        for _ in range(40):
            x = float(cfg.start_xy[0] + self._rng.uniform(-cfg.start_noise, cfg.start_noise))
            y = float(cfg.start_xy[1] + self._rng.uniform(-cfg.start_noise, cfg.start_noise))
            x = float(np.clip(x, cfg.uav_radius, cfg.map_width - cfg.uav_radius))
            y = float(np.clip(y, cfg.uav_radius, cfg.map_height - cfg.uav_radius))
            if self._point_clear(x, y):
                return _UAV(x, y, cfg.start_speed, 0.0)
        return _UAV(float(cfg.start_xy[0]), float(cfg.start_xy[1]), cfg.start_speed, 0.0)

    def _point_clear(self, x: float, y: float) -> bool:
        cfg = self.cfg
        if x < cfg.uav_radius or y < cfg.uav_radius:
            return False
        if x > cfg.map_width - cfg.uav_radius or y > cfg.map_height - cfg.uav_radius:
            return False
        for o in self._obstacle_init:
            if math.hypot(x - o.x, y - o.y) < o.radius + cfg.uav_radius + 8.0:
                return False
        for u in self.uavs:
            if math.hypot(x - u.x, y - u.y) < 2.5 * cfg.uav_radius:
                return False
        return True

    def _deliver_initial_leader_state(self) -> None:
        """Free delivery at reset. Not counted in the packet-success rate."""
        leader = self.uavs[0]
        self.links = [
            _Link(leader.x, leader.y, leader.v, leader.heading, 0, 0)
            for _ in range(self.n_followers)
        ]

    def _integrate(self, actions: np.ndarray) -> None:
        cfg = self.cfg
        for u, a in zip(self.uavs, actions):
            accel = float(a[0]) * cfg.max_accel
            yaw_rate = float(a[1]) * cfg.max_yaw_rate
            u.v = float(np.clip(u.v + accel * cfg.dt, cfg.v_min, cfg.v_max))
            u.heading = wrap_angle(u.heading + yaw_rate * cfg.dt)
            u.x += (u.v * math.cos(u.heading) + self.wind_x) * cfg.dt
            u.y += (u.v * math.sin(u.heading) + self.wind_y) * cfg.dt

    def _move_obstacles(self) -> None:
        cfg = self.cfg
        for o in self.obstacles:
            if o.vx == 0.0 and o.vy == 0.0:
                continue
            o.x += o.vx * cfg.dt
            o.y += o.vy * cfg.dt
            if o.x < o.radius or o.x > cfg.map_width - o.radius:
                o.vx *= -1.0
                o.x = float(np.clip(o.x, o.radius, cfg.map_width - o.radius))
            if o.y < o.radius or o.y > cfg.map_height - o.radius:
                o.vy *= -1.0
                o.y = float(np.clip(o.y, o.radius, cfg.map_height - o.radius))

    def _apply_bounds(self) -> np.ndarray:
        cfg = self.cfg
        oob = np.zeros(self.n_agents, dtype=bool)
        for i, u in enumerate(self.uavs):
            cx = float(np.clip(u.x, cfg.uav_radius, cfg.map_width - cfg.uav_radius))
            cy = float(np.clip(u.y, cfg.uav_radius, cfg.map_height - cfg.uav_radius))
            if cx != u.x or cy != u.y:
                oob[i] = True
                u.x, u.y = cx, cy
        return oob

    def _collision_flags(self) -> tuple[bool, bool]:
        cfg = self.cfg
        hit_obstacle = False
        for u in self.uavs:
            for o in self.obstacles:
                gap = math.hypot(u.x - o.x, u.y - o.y) - o.radius - cfg.uav_radius
                self._min_clearance = min(self._min_clearance, gap)
                if gap < 0.0:
                    hit_obstacle = True
        hit_uav = False
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                a = self.uavs[i]
                b = self.uavs[j]
                if math.hypot(a.x - b.x, a.y - b.y) < 2.0 * cfg.uav_radius:
                    hit_uav = True
        return hit_obstacle, hit_uav

    def _advance_waypoints(self) -> int:
        """Capture the current waypoint if the leader is inside ``wp_radius``.

        Returns how many waypoints were captured on this step (0 or 1; a fast
        step can chain if several waypoints overlap, which the default route
        does not do).
        """
        captured = 0
        while self._wp_done < len(self.waypoints) and self._waypoint_distance() <= self.cfg.wp_radius:
            self._wp_done += 1
            captured += 1
            if self._wp_index < len(self.waypoints) - 1:
                self._wp_index += 1
            else:
                break
        return captured

    def _segment_blocked(self, ax: float, ay: float, bx: float, by: float) -> bool:
        margin = float(self.cfg.blockage_margin)
        for o in self.obstacles:
            if segment_hits_disk(ax, ay, bx, by, o.x, o.y, o.radius + margin):
                return True
        return False

    def _update_channel(self) -> np.ndarray:
        leader = self.uavs[0]
        ok = np.zeros(self.n_followers, dtype=np.float32)
        for i, link in enumerate(self.links):
            follower = self.uavs[i + 1]
            dist = math.hypot(follower.x - leader.x, follower.y - leader.y)
            blocked = self._segment_blocked(leader.x, leader.y, follower.x, follower.y)
            link.blocked = int(blocked)
            if blocked:
                self._nlos_count += 1
            apply_block = bool(self.ep_use_blockage and self.ep_channel_mode == "distance" and blocked)
            p = packet_success_probability(
                dist,
                self.ep_p0,
                self.ep_d0,
                self.ep_channel_mode,
                blocked=apply_block,
                nlos_factor=float(self.cfg.nlos_factor),
            )
            success = bool(self._rng.random() < p)
            if success:
                if self._outage_streak[i] > 0:
                    self._outage_lengths.append(int(self._outage_streak[i]))
                    self._outage_streak[i] = 0
                link.last_x = leader.x
                link.last_y = leader.y
                link.last_v = leader.v
                link.last_heading = leader.heading
                link.aoi = 0
                ok[i] = 1.0
            else:
                link.aoi = int(min(link.aoi + 1, self.cfg.aoi_cap))
                self._outage_streak[i] += 1
                ok[i] = 0.0
            self._aoi_sum += link.aoi
            self._aoi_max = max(self._aoi_max, link.aoi)
        self._packets_ok += int(ok.sum())
        self._packets_attempted += self.n_followers
        return ok

    def _reference_pose(self, follower_index: int) -> np.ndarray:
        link = self.links[follower_index]
        leader = self.uavs[0]
        if self.ep_obs_mode == "perfect":
            return np.array([leader.x, leader.y, leader.v, leader.heading], dtype=np.float64)
        if self.ep_obs_mode == "extrapolate":
            age = float(link.aoi)
            x = link.last_x + link.last_v * math.cos(link.last_heading) * self.cfg.dt * age
            y = link.last_y + link.last_v * math.sin(link.last_heading) * self.cfg.dt * age
            return np.array([x, y, link.last_v, link.last_heading], dtype=np.float64)
        return np.array(
            [link.last_x, link.last_y, link.last_v, link.last_heading], dtype=np.float64
        )

    def _observations(self) -> list[np.ndarray]:
        cfg = self.cfg
        wind_obs = np.array(
            [
                self.wind_x + float(self._rng.normal(0.0, cfg.wind_obs_noise)),
                self.wind_y + float(self._rng.normal(0.0, cfg.wind_obs_noise)),
            ],
            dtype=np.float64,
        )
        obs_list: list[np.ndarray] = []
        for i, u in enumerate(self.uavs):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            obs[ObsIndex.X] = u.x / cfg.map_width
            obs[ObsIndex.Y] = u.y / cfg.map_height
            obs[ObsIndex.V] = u.v / cfg.v_max
            obs[ObsIndex.SIN_H] = math.sin(u.heading)
            obs[ObsIndex.COS_H] = math.cos(u.heading)
            obs[ObsIndex.WIND_X] = wind_obs[0] / cfg.wind_max
            obs[ObsIndex.WIND_Y] = wind_obs[1] / cfg.wind_max
            if i == 0:
                ref_x, ref_y = self.waypoints[self._wp_index]
                ref_v = 0.0
                ref_h = math.atan2(ref_y - u.y, ref_x - u.x)
                aoi_norm = 0.0
                slot_x, slot_y = 0.0, 0.0
                des_x, des_y = ref_x, ref_y
                obs[ObsIndex.IS_LEADER] = 1.0
                obs[ObsIndex.WP_FRAC] = self._wp_done / len(self.waypoints)
            else:
                ref = self._reference_pose(i - 1)
                ref_x, ref_y, ref_v, ref_h = (
                    float(ref[0]),
                    float(ref[1]),
                    float(ref[2]),
                    float(ref[3]),
                )
                show_aoi = self.ep_obs_mode == "aoi" and self.ep_use_aoi
                aoi_norm = (self.links[i - 1].aoi / cfg.aoi_cap) if show_aoi else 0.0
                slot_x, slot_y = self.offsets[i - 1]
                des = desired_slot((ref_x, ref_y), ref_h, (slot_x, slot_y))
                des_x, des_y = float(des[0]), float(des[1])
                obs[ObsIndex.IS_LEADER] = 0.0
                # Waypoint progress is the leader's mission. Followers do not
                # get it for free; they only see the leader through the link.
                obs[ObsIndex.WP_FRAC] = 0.0
            obs[ObsIndex.REF_REL_X] = (ref_x - u.x) / cfg.map_width
            obs[ObsIndex.REF_REL_Y] = (ref_y - u.y) / cfg.map_height
            obs[ObsIndex.REF_V] = ref_v / cfg.v_max
            obs[ObsIndex.REF_SIN] = math.sin(ref_h)
            obs[ObsIndex.REF_COS] = math.cos(ref_h)
            obs[ObsIndex.AOI] = aoi_norm
            obs[ObsIndex.SLOT_X] = slot_x / cfg.formation_scale
            obs[ObsIndex.SLOT_Y] = slot_y / cfg.formation_scale
            obs[ObsIndex.DES_REL_X] = (des_x - u.x) / cfg.map_width
            obs[ObsIndex.DES_REL_Y] = (des_y - u.y) / cfg.map_height
            self._write_teammate(obs, i)
            self._write_obstacles(obs, u)
            obs_list.append(obs)
        return obs_list

    def _write_teammate(self, obs: np.ndarray, index: int) -> None:
        cfg = self.cfg
        origin = self.uavs[index]
        best = None
        best_d = float("inf")
        for j, other in enumerate(self.uavs):
            if j == index:
                continue
            d = math.hypot(other.x - origin.x, other.y - origin.y)
            if d < best_d:
                best_d = d
                best = other
        assert best is not None
        if best_d <= cfg.sensor_range:
            obs[ObsIndex.TEAMMATE_REL_X] = (best.x - origin.x) / cfg.map_width
            obs[ObsIndex.TEAMMATE_REL_Y] = (best.y - origin.y) / cfg.map_height
            obs[ObsIndex.TEAMMATE_DIST] = best_d / self.diag
            obs[ObsIndex.TEAMMATE_PRESENT] = 1.0

    def _write_obstacles(self, obs: np.ndarray, uav: _UAV) -> None:
        cfg = self.cfg
        ranked = []
        for o in self.obstacles:
            d = math.hypot(o.x - uav.x, o.y - uav.y)
            if d <= cfg.sensor_range:
                ranked.append((d, o))
        ranked.sort(key=lambda item: item[0])
        for k in range(cfg.n_nearest_obstacles):
            base = ObsIndex.OBS_START + k * ObsIndex.OBS_FEATS
            if k < len(ranked):
                o = ranked[k][1]
                obs[base + 0] = (o.x - uav.x) / cfg.map_width
                obs[base + 1] = (o.y - uav.y) / cfg.map_height
                obs[base + 2] = o.vx / cfg.obstacle_speed_norm
                obs[base + 3] = o.vy / cfg.obstacle_speed_norm
                obs[base + 4] = o.radius / self.diag
                obs[base + 5] = 1.0

    def _true_slots(self) -> np.ndarray:
        leader = self.uavs[0]
        return np.stack(
            [desired_slot((leader.x, leader.y), leader.heading, off) for off in self.offsets],
            axis=0,
        )

    def _formation_errors(self) -> np.ndarray:
        slots = self._true_slots()
        errors = np.zeros(self.n_followers, dtype=np.float64)
        for i in range(self.n_followers):
            u = self.uavs[i + 1]
            errors[i] = math.hypot(u.x - slots[i, 0], u.y - slots[i, 1])
        return errors

    def _waypoint_distance(self) -> float:
        leader = self.uavs[0]
        wx, wy = self.waypoints[self._wp_index]
        return math.hypot(leader.x - wx, leader.y - wy)

    def _nearest_gap(self, uav: _UAV) -> float:
        if not self.obstacles:
            return float("inf")
        cfg = self.cfg
        return min(
            math.hypot(uav.x - o.x, uav.y - o.y) - o.radius - cfg.uav_radius
            for o in self.obstacles
        )

    def _rewards(
        self,
        actions: np.ndarray,
        errors: np.ndarray,
        wp_dist: float,
        captured: int,
        hit_obstacle: bool,
        hit_uav: bool,
        oob: np.ndarray,
    ) -> np.ndarray:
        """Per-agent dense reward. Formulas are in the README.

        Collision, waypoint capture, and success are shared by the team.
        Clearance and boundary penalties are individual.
        """
        cfg = self.cfg
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        leader = self.uavs[0]
        wx, wy = self.waypoints[self._wp_index]
        bearing = math.atan2(wy - leader.y, wx - leader.x)
        smooth0 = float(actions[0, 0] ** 2 + actions[0, 1] ** 2)
        rewards[0] = (
            cfg.w_progress * (self._prev_wp_dist - wp_dist)
            + cfg.w_heading * math.cos(wrap_angle(bearing - leader.heading))
            - cfg.w_smooth * smooth0
            - cfg.w_step
        )
        for i in range(self.n_followers):
            smooth = float(actions[i + 1, 0] ** 2 + actions[i + 1, 1] ** 2)
            align = math.cos(wrap_angle(leader.heading - self.uavs[i + 1].heading))
            bonus = cfg.formation_bonus if errors[i] <= cfg.formation_tol else 0.0
            rewards[i + 1] = (
                cfg.w_form_progress * (self._prev_form_err[i] - errors[i])
                - cfg.w_formation * errors[i]
                + cfg.w_heading * align
                + bonus
                - cfg.w_smooth * smooth
                - cfg.w_step
            )
        for i, u in enumerate(self.uavs):
            gap = self._nearest_gap(u)
            if gap < cfg.clearance_margin:
                rewards[i] -= np.float32(
                    cfg.w_clearance * (cfg.clearance_margin - gap) / cfg.clearance_margin
                )
        if captured:
            rewards += np.float32(cfg.wp_bonus * captured)
        if hit_obstacle or hit_uav:
            rewards -= np.float32(cfg.collision_penalty)
        route_done = self._wp_done >= len(self.waypoints)
        if route_done and np.all(errors <= cfg.formation_tol) and not (hit_obstacle or hit_uav):
            rewards += np.float32(cfg.success_bonus)
        rewards -= oob.astype(np.float32) * np.float32(cfg.oob_penalty)
        return rewards

    def _reset_stat_counters(self) -> None:
        self._form_keep_steps = 0
        self._form_err_sum = 0.0
        self._aoi_sum = 0.0
        self._aoi_max = 0
        self._packets_ok = 0
        self._packets_attempted = 0
        self._nlos_count = 0
        self._outage_streak = np.zeros(self.n_followers, dtype=np.int32)
        self._outage_lengths: list[int] = []
        self._stat_steps = 0
        self._min_clearance = float("inf")

    def _accumulate(self, errors: np.ndarray, in_formation: bool) -> None:
        self._stat_steps += 1
        self._form_err_sum += float(errors.mean())
        if in_formation:
            self._form_keep_steps += 1

    def _episode_dict(
        self,
        success: bool,
        collided: bool,
        timeout: bool,
        hit_obstacle: bool,
        hit_uav: bool,
    ) -> dict:
        lengths = list(self._outage_lengths)
        for streak in self._outage_streak:
            if int(streak) > 0:
                lengths.append(int(streak))
        steps = max(self._stat_steps, 1)
        clearance = self._min_clearance
        if clearance == float("inf"):
            clearance = 0.0
        return {
            "success": float(success),
            "collision": float(collided),
            "obstacle_collision": float(hit_obstacle),
            "inter_uav_collision": float(hit_uav),
            "timeout": float(timeout),
            "formation_keeping_rate": self._form_keep_steps / steps,
            "mean_formation_error": self._form_err_sum / steps,
            "mean_aoi": self._aoi_sum / (steps * self.n_followers),
            "max_aoi": float(self._aoi_max),
            "packet_success_rate": (
                self._packets_ok / self._packets_attempted if self._packets_attempted else 1.0
            ),
            "mean_outage_length": float(np.mean(lengths)) if lengths else 0.0,
            "waypoint_fraction": self._wp_done / len(self.waypoints),
            "nlos_fraction": (
                self._nlos_count / self._packets_attempted if self._packets_attempted else 0.0
            ),
            "min_clearance": float(clearance),
            "steps": int(self._step_count),
            "p0": float(self.ep_p0),
        }

    def _info_skeleton(self) -> dict:
        return {
            "p0": float(self.ep_p0),
            "channel_mode": self.ep_channel_mode,
            "obs_mode": self.ep_obs_mode,
            "use_aoi": bool(self.ep_use_aoi),
            "use_blockage": bool(self.ep_use_blockage),
            "aoi": np.array([link.aoi for link in self.links], dtype=np.int32),
            "wind": np.array([self.wind_x, self.wind_y], dtype=np.float32),
            "waypoints_done": int(self._wp_done),
            "collision": False,
            "success": False,
            "episode": None,
        }
