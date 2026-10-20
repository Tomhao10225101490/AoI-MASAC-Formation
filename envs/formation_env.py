"""2D leader-follower UAV formation with a lossy leader link and AoI.

This environment is a gym-style simulator. It does not depend on Gymnasium.

    obs, info = env.reset(seed=None, options=None)
    obs, rewards, terminated, truncated, info = env.step(actions)

``obs`` is a list of length ``n_agents`` (leader first). ``rewards`` has shape
``(n_agents,)``. ``terminated`` is True on success or collision. ``truncated``
is True when ``max_steps`` is reached without either of those.

Modeling assumptions (read this before changing the code)
---------------------------------------------------------
* One leader and ``n_followers`` followers fly in a rectangle (default
  700 m by 600 m). Motion is a simple unicycle: each step the agent changes
  its speed and its heading, then moves.
* Only the leader's state (position, speed, heading) is sent over the lossy
  link. Each follower draws an independent Bernoulli packet with

      p_succ = p0 * exp(-d / d0)     (channel_mode="distance")

  where ``d`` is that follower's distance to the leader *after* everyone has
  moved this step. On success the follower stores the leader's new state and
  AoI resets to 0. On failure it keeps the previous state and AoI increments
  by 1, capped at ``aoi_cap``.
* ``channel_mode="independent"`` uses ``p_succ = p0`` at every distance
  (the distance-independent ablation). ``channel_mode="perfect"`` always
  delivers.
* If ``randomize_channel`` is set, ``p0`` is drawn uniformly from
  ``[p0_low, p0_high]`` at the start of each episode (not when mode is
  ``perfect``). A value passed through ``reset(..., options={"p0": ...})``
  wins over the draw, which is how the packet-loss sweep fixes the channel.
* Obstacle positions and the nearest other UAV are *onboard sensing*. They
  are not sent on the lossy leader link. The thing that goes stale is the
  leader state, because that is the information the formation controller
  actually has to receive.
* The reward uses the true leader pose. The follower's observation of
  "where my slot is" uses only the leader pose it is allowed to know
  (fresh, stale, or linearly extrapolated). That gap is the task.
* The centralized critic is not part of this file. Training code builds a
  privileged ``global_state()`` that includes true poses and every
  follower's AoI. Actors only see ``obs``.

Observation layout (same length for every agent)
-------------------------------------------------
See ``ObsIndex``. Leader entries that do not apply (AoI, slot) are zero,
and the leader's reference point is the goal instead of a leader message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from envs.channel import packet_success_probability
from envs.geometry import desired_slot, make_formation_offsets, wrap_angle


class ObsIndex:
    """Column indices of one agent's observation vector."""

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
    TEAMMATE_REL_X = 15
    TEAMMATE_REL_Y = 16
    TEAMMATE_DIST = 17
    IS_LEADER = 18
    OBS_START = 19
    OBS_FEATS = 4  # rel_x, rel_y, radius_norm, present

    @staticmethod
    def dim(n_nearest_obstacles: int) -> int:
        return ObsIndex.OBS_START + ObsIndex.OBS_FEATS * int(n_nearest_obstacles)


def global_state_dim(n_agents: int, n_followers: int, max_obstacles: int) -> int:
    """Length of the privileged critic state. Fixed for a given env config."""
    return 5 * n_agents + 2 + 5 * max_obstacles + 6 * n_followers


@dataclass
class EnvConfig:
    """All environment knobs. Unknown YAML keys are rejected by the loader."""

    map_width: float = 700.0
    map_height: float = 600.0
    n_followers: int = 4
    dt: float = 1.0
    max_steps: int = 250
    v_min: float = 0.0
    v_max: float = 6.0
    max_speed_delta: float = 1.5
    max_heading_delta: float = 0.40
    uav_radius: float = 12.0
    goal_radius: float = 40.0
    formation_tol: float = 45.0
    formation_scale: float = 120.0
    n_nearest_obstacles: int = 1
    max_obstacles: int = 1
    obstacles: list = field(
        default_factory=lambda: [
            {"x": 420.0, "y": 300.0, "radius": 40.0, "vx": 0.0, "vy": 0.0}
        ]
    )
    formation_offsets: list | None = None
    channel_mode: str = "distance"
    p0: float = 0.9
    d0: float = 250.0
    randomize_channel: bool = True
    p0_low: float = 0.3
    p0_high: float = 1.0
    aoi_cap: int = 30
    obs_mode: str = "aoi"  # aoi | perfect | hold | extrapolate
    use_aoi: bool = True
    goal_xy: tuple = (640.0, 300.0)
    start_xy: tuple = (200.0, 300.0)
    start_speed: float = 2.0
    start_noise: float = 15.0
    seed: int = 0
    # Reward weights. These are a starting point, not a tuned paper result.
    w_progress: float = 0.05
    w_heading: float = 0.10
    w_formation: float = 0.01
    w_form_progress: float = 0.05
    w_smooth: float = 0.02
    w_step: float = 0.02
    formation_bonus: float = 0.20
    collision_penalty: float = 40.0
    success_bonus: float = 50.0
    oob_penalty: float = 2.0
    obstacle_speed_norm: float = 5.0

    def __post_init__(self) -> None:
        self.goal_xy = (float(self.goal_xy[0]), float(self.goal_xy[1]))
        self.start_xy = (float(self.start_xy[0]), float(self.start_xy[1]))
        if self.n_followers < 1:
            raise ValueError("n_followers must be >= 1")
        if self.n_nearest_obstacles < 1:
            raise ValueError("n_nearest_obstacles must be >= 1")
        if self.max_obstacles < 0:
            raise ValueError("max_obstacles must be >= 0")
        if len(self.obstacles) > self.max_obstacles:
            raise ValueError(
                f"{len(self.obstacles)} obstacles were given but max_obstacles="
                f"{self.max_obstacles}. Raise max_obstacles (this changes the "
                "critic input size, so it has to match training)."
            )
        if self.obs_mode not in ("aoi", "perfect", "hold", "extrapolate"):
            raise ValueError(f"bad obs_mode {self.obs_mode!r}")
        if self.channel_mode not in ("distance", "independent", "perfect"):
            raise ValueError(f"bad channel_mode {self.channel_mode!r}")
        if self.aoi_cap < 1:
            raise ValueError("aoi_cap must be >= 1")
        if self.formation_offsets is not None and len(self.formation_offsets) != self.n_followers:
            raise ValueError("formation_offsets length must equal n_followers")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.v_max < self.v_min:
            raise ValueError("v_max must be >= v_min")
        if self.map_width <= 0.0 or self.map_height <= 0.0:
            raise ValueError("map size must be positive")


def env_config_from_dict(data: dict) -> EnvConfig:
    """Build an ``EnvConfig`` and reject typo keys instead of ignoring them."""
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
    """What one follower currently believes about the leader."""

    last_x: float
    last_y: float
    last_v: float
    last_heading: float
    aoi: int


class UAVFormationEnv:
    """Leader-follower formation on a rectangle with a lossy leader link."""

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config if config is not None else EnvConfig()
        self.n_followers = int(self.cfg.n_followers)
        self.n_agents = self.n_followers + 1
        self.action_dim = 2
        self.obs_dim = ObsIndex.dim(self.cfg.n_nearest_obstacles)
        self.obs_dims = [self.obs_dim] * self.n_agents
        self.global_dim = global_state_dim(
            self.n_agents, self.n_followers, int(self.cfg.max_obstacles)
        )
        if self.cfg.formation_offsets is None:
            self.offsets = make_formation_offsets(self.n_followers)
        else:
            self.offsets = [
                (float(p[0]), float(p[1])) for p in self.cfg.formation_offsets
            ]
        self._obstacle_init = [_Obstacle.from_dict(o) for o in self.cfg.obstacles]
        self.diag = float(math.hypot(self.cfg.map_width, self.cfg.map_height))
        self._rng = np.random.default_rng(self.cfg.seed)
        self.uavs: list[_UAV] = []
        self.obstacles: list[_Obstacle] = []
        self.links: list[_Link] = []
        self._step_count = 0
        self._prev_goal_dist = 0.0
        self._prev_form_err = np.zeros(self.n_followers, dtype=np.float64)
        self.ep_p0 = float(self.cfg.p0)
        self.ep_channel_mode = self.cfg.channel_mode
        self.ep_obs_mode = self.cfg.obs_mode
        self.ep_use_aoi = bool(self.cfg.use_aoi)
        self._reset_stat_counters()

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        """Reset the episode.

        ``options`` may set ``p0``, ``channel_mode``, ``obs_mode``,
        ``use_aoi``, ``randomize_channel``, or ``d0`` for this episode only.
        Passing ``seed`` rebuilds the RNG; omitting it continues the stream.
        """
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._apply_options(options or {})
        self._spawn()
        self._deliver_initial_leader_state()
        self._step_count = 0
        self._reset_stat_counters()
        self._prev_goal_dist = self._goal_distance()
        self._prev_form_err = self._formation_errors()
        obs = self._observations()
        info = self._info_skeleton()
        return obs, info

    def step(
        self, actions: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray, bool, bool, dict]:
        """Advance one step. ``actions`` has shape ``(n_agents, 2)`` in [-1, 1]."""
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
        packet_ok = self._update_channel()
        errors = self._formation_errors()
        goal_dist = self._goal_distance()
        rewards = self._rewards(
            actions=actions,
            errors=errors,
            goal_dist=goal_dist,
            hit_obstacle=hit_obstacle,
            hit_uav=hit_uav,
            oob=oob,
        )

        self._step_count += 1
        collided = bool(hit_obstacle or hit_uav)
        in_formation = bool(np.all(errors <= self.cfg.formation_tol))
        at_goal = bool(goal_dist <= self.cfg.goal_radius)
        # A collision on the same step as arrival counts as a failure.
        success = bool(at_goal and in_formation and not collided)
        timeout = bool(self._step_count >= self.cfg.max_steps and not success and not collided)
        terminated = bool(success or collided)
        truncated = bool(timeout)

        self._accumulate(
            errors=errors,
            in_formation=in_formation,
            packet_ok=packet_ok,
            hit_obstacle=hit_obstacle,
            hit_uav=hit_uav,
            success=success,
            timeout=timeout,
        )
        self._prev_goal_dist = goal_dist
        self._prev_form_err = errors

        obs = self._observations()
        info = self._info_skeleton()
        info["collision"] = collided
        info["success"] = success
        info["packet_success"] = packet_ok.copy()
        info["formation_errors"] = errors.copy()
        if terminated or truncated:
            info["episode"] = self._episode_dict(success, collided, timeout, hit_obstacle, hit_uav)
        return obs, rewards, terminated, truncated, info

    def global_state(self) -> np.ndarray:
        """Privileged state for the centralized critic. Not given to actors."""
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
        parts.extend([cfg.goal_xy[0] / cfg.map_width, cfg.goal_xy[1] / cfg.map_height])
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
                ]
            )
        state = np.asarray(parts, dtype=np.float32)
        if state.shape != (self.global_dim,):
            raise RuntimeError(
                f"global state length {state.shape[0]} != {self.global_dim}"
            )
        return state

    def reference_leader_state(self, follower_index: int) -> np.ndarray:
        """Leader ``(x, y, v, heading)`` actually written into that follower's obs.

        This is the fresh, held, or extrapolated pose, depending on
        ``ep_obs_mode``. It is *not* always the true leader pose.
        """
        return self._reference_pose(follower_index)

    def snapshot(self) -> dict:
        """Lightweight state dump for trajectory plots."""
        return {
            "uav_xy": np.array([[u.x, u.y] for u in self.uavs], dtype=np.float32),
            "heading": np.array([u.heading for u in self.uavs], dtype=np.float32),
            "obstacle_xy": np.array(
                [[o.x, o.y] for o in self.obstacles], dtype=np.float32
            ).reshape(-1, 2),
            "obstacle_r": np.array([o.radius for o in self.obstacles], dtype=np.float32),
            "aoi": np.array([link.aoi for link in self.links], dtype=np.float32),
            "goal": np.array(self.cfg.goal_xy, dtype=np.float32),
            "slots": self._true_slots().astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Channel and observation internals
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
        if p0 < 0.0 or p0 > 1.0:
            raise ValueError(f"p0 must be in [0, 1], got {p0}")
        self.ep_channel_mode = mode
        self.ep_p0 = p0
        self.ep_d0 = float(options.get("d0", cfg.d0))
        self.ep_obs_mode = obs_mode
        if "use_aoi" in options and options["use_aoi"] is not None:
            self.ep_use_aoi = bool(options["use_aoi"])
        else:
            self.ep_use_aoi = bool(cfg.use_aoi)

    def _spawn(self) -> None:
        cfg = self.cfg
        leader = self._sample_leader_pose()
        self.uavs = [leader]
        for offset in self.offsets:
            placed = False
            slot = desired_slot((leader.x, leader.y), leader.heading, offset)
            for _ in range(40):
                noise = self._rng.uniform(-cfg.start_noise, cfg.start_noise, size=2)
                x = float(slot[0] + noise[0])
                y = float(slot[1] + noise[1])
                heading = wrap_angle(leader.heading + float(self._rng.uniform(-0.1, 0.1)))
                if self._point_clear(x, y, ignore_agents=False):
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
            if self._point_clear(x, y, ignore_agents=True):
                return _UAV(x, y, cfg.start_speed, 0.0)
        return _UAV(float(cfg.start_xy[0]), float(cfg.start_xy[1]), cfg.start_speed, 0.0)

    def _point_clear(self, x: float, y: float, ignore_agents: bool) -> bool:
        cfg = self.cfg
        if x < cfg.uav_radius or y < cfg.uav_radius:
            return False
        if x > cfg.map_width - cfg.uav_radius or y > cfg.map_height - cfg.uav_radius:
            return False
        for o in self._obstacle_init:
            if math.hypot(x - o.x, y - o.y) < o.radius + cfg.uav_radius + 5.0:
                return False
        if not ignore_agents:
            for u in self.uavs:
                if math.hypot(x - u.x, y - u.y) < 2.0 * cfg.uav_radius:
                    return False
        return True

    def _deliver_initial_leader_state(self) -> None:
        """At reset every follower starts informed (AoI = 0).

        This delivery is not counted in the packet-success rate, so the rate
        describes only the in-episode channel.
        """
        leader = self.uavs[0]
        self.links = [
            _Link(leader.x, leader.y, leader.v, leader.heading, 0)
            for _ in range(self.n_followers)
        ]

    def _integrate(self, actions: np.ndarray) -> None:
        cfg = self.cfg
        for u, a in zip(self.uavs, actions):
            u.v = float(np.clip(u.v + a[0] * cfg.max_speed_delta, cfg.v_min, cfg.v_max))
            u.heading = wrap_angle(u.heading + a[1] * cfg.max_heading_delta)
            u.x += u.v * math.cos(u.heading) * cfg.dt
            u.y += u.v * math.sin(u.heading) * cfg.dt

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
        """Clip UAVs into the map. Returns a boolean per agent, True if clipped."""
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
                if math.hypot(u.x - o.x, u.y - o.y) < o.radius + cfg.uav_radius:
                    hit_obstacle = True
                    break
            if hit_obstacle:
                break
        hit_uav = False
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                a = self.uavs[i]
                b = self.uavs[j]
                if math.hypot(a.x - b.x, a.y - b.y) < 2.0 * cfg.uav_radius:
                    hit_uav = True
                    break
            if hit_uav:
                break
        return hit_obstacle, hit_uav

    def _update_channel(self) -> np.ndarray:
        """Draw one packet per follower. Update AoI and the stored leader state."""
        leader = self.uavs[0]
        ok = np.zeros(self.n_followers, dtype=np.float32)
        for i, link in enumerate(self.links):
            follower = self.uavs[i + 1]
            dist = math.hypot(follower.x - leader.x, follower.y - leader.y)
            p = packet_success_probability(
                dist, self.ep_p0, self.ep_d0, self.ep_channel_mode
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
        """Pose written into the follower observation under the active obs mode."""
        link = self.links[follower_index]
        leader = self.uavs[0]
        mode = self.ep_obs_mode
        if mode == "perfect":
            return np.array(
                [leader.x, leader.y, leader.v, leader.heading], dtype=np.float64
            )
        if mode == "extrapolate":
            age = float(link.aoi)
            x = link.last_x + link.last_v * math.cos(link.last_heading) * self.cfg.dt * age
            y = link.last_y + link.last_v * math.sin(link.last_heading) * self.cfg.dt * age
            return np.array([x, y, link.last_v, link.last_heading], dtype=np.float64)
        # "aoi" and "hold" both reuse the last received pose. They differ only
        # in whether the AoI feature is filled in (see _observations).
        return np.array(
            [link.last_x, link.last_y, link.last_v, link.last_heading], dtype=np.float64
        )

    def _observations(self) -> list[np.ndarray]:
        cfg = self.cfg
        obs_list: list[np.ndarray] = []
        leader = self.uavs[0]
        for i, u in enumerate(self.uavs):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            obs[ObsIndex.X] = u.x / cfg.map_width
            obs[ObsIndex.Y] = u.y / cfg.map_height
            obs[ObsIndex.V] = u.v / cfg.v_max
            obs[ObsIndex.SIN_H] = math.sin(u.heading)
            obs[ObsIndex.COS_H] = math.cos(u.heading)
            if i == 0:
                ref_x, ref_y = cfg.goal_xy
                ref_v = 0.0
                bearing = math.atan2(ref_y - u.y, ref_x - u.x)
                ref_h = bearing
                aoi_norm = 0.0
                slot_x, slot_y = 0.0, 0.0
                des_x, des_y = ref_x, ref_y
                obs[ObsIndex.IS_LEADER] = 1.0
            else:
                ref = self._reference_pose(i - 1)
                ref_x, ref_y, ref_v, ref_h = (float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3]))
                show_aoi = self.ep_obs_mode == "aoi" and self.ep_use_aoi
                aoi_norm = (self.links[i - 1].aoi / cfg.aoi_cap) if show_aoi else 0.0
                slot_x, slot_y = self.offsets[i - 1]
                des = desired_slot((ref_x, ref_y), ref_h, (slot_x, slot_y))
                des_x, des_y = float(des[0]), float(des[1])
                obs[ObsIndex.IS_LEADER] = 0.0
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
            mate = self._nearest_teammate(i)
            obs[ObsIndex.TEAMMATE_REL_X] = (mate.x - u.x) / cfg.map_width
            obs[ObsIndex.TEAMMATE_REL_Y] = (mate.y - u.y) / cfg.map_height
            obs[ObsIndex.TEAMMATE_DIST] = math.hypot(mate.x - u.x, mate.y - u.y) / self.diag
            self._write_obstacles(obs, u)
            obs_list.append(obs)
        return obs_list

    def _nearest_teammate(self, index: int) -> _UAV:
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
        assert best is not None  # n_followers >= 1 guarantees a teammate
        return best

    def _write_obstacles(self, obs: np.ndarray, uav: _UAV) -> None:
        cfg = self.cfg
        ranked = []
        for o in self.obstacles:
            d = math.hypot(o.x - uav.x, o.y - uav.y)
            ranked.append((d, o))
        ranked.sort(key=lambda item: item[0])
        for k in range(cfg.n_nearest_obstacles):
            base = ObsIndex.OBS_START + k * ObsIndex.OBS_FEATS
            if k < len(ranked):
                o = ranked[k][1]
                obs[base + 0] = (o.x - uav.x) / cfg.map_width
                obs[base + 1] = (o.y - uav.y) / cfg.map_height
                obs[base + 2] = o.radius / self.diag
                obs[base + 3] = 1.0
            else:
                obs[base + 3] = 0.0

    def _true_slots(self) -> np.ndarray:
        leader = self.uavs[0]
        slots = [
            desired_slot((leader.x, leader.y), leader.heading, off) for off in self.offsets
        ]
        return np.stack(slots, axis=0)

    def _formation_errors(self) -> np.ndarray:
        slots = self._true_slots()
        errors = np.zeros(self.n_followers, dtype=np.float64)
        for i in range(self.n_followers):
            u = self.uavs[i + 1]
            errors[i] = math.hypot(u.x - slots[i, 0], u.y - slots[i, 1])
        return errors

    def _goal_distance(self) -> float:
        leader = self.uavs[0]
        return math.hypot(leader.x - self.cfg.goal_xy[0], leader.y - self.cfg.goal_xy[1])

    def _rewards(
        self,
        actions: np.ndarray,
        errors: np.ndarray,
        goal_dist: float,
        hit_obstacle: bool,
        hit_uav: bool,
        oob: np.ndarray,
    ) -> np.ndarray:
        """Per-agent dense reward. Formulas are written out in the README.

        Collision and success are team events: every agent receives the same
        penalty or bonus. Out-of-bounds is individual because only the agent
        that hit the border is clipped.
        """
        cfg = self.cfg
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        leader = self.uavs[0]
        bearing = math.atan2(cfg.goal_xy[1] - leader.y, cfg.goal_xy[0] - leader.x)
        heading_align = math.cos(wrap_angle(bearing - leader.heading))
        smooth0 = float(actions[0, 0] ** 2 + actions[0, 1] ** 2)
        rewards[0] = (
            cfg.w_progress * (self._prev_goal_dist - goal_dist)
            + cfg.w_heading * heading_align
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
        if hit_obstacle or hit_uav:
            rewards -= np.float32(cfg.collision_penalty)
        if goal_dist <= cfg.goal_radius and np.all(errors <= cfg.formation_tol) and not (
            hit_obstacle or hit_uav
        ):
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
        self._outage_streak = np.zeros(self.n_followers, dtype=np.int32)
        self._outage_lengths: list[int] = []
        self._stat_steps = 0

    def _accumulate(
        self,
        errors: np.ndarray,
        in_formation: bool,
        packet_ok: np.ndarray,
        hit_obstacle: bool,
        hit_uav: bool,
        success: bool,
        timeout: bool,
    ) -> None:
        del packet_ok, hit_obstacle, hit_uav, success, timeout  # recorded elsewhere
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
        # Close any outage that was still running when the episode ended.
        lengths = list(self._outage_lengths)
        for streak in self._outage_streak:
            if streak > 0:
                lengths.append(int(streak))
        steps = max(self._stat_steps, 1)
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
            "steps": int(self._step_count),
            "p0": float(self.ep_p0),
        }

    def _info_skeleton(self) -> dict:
        return {
            "p0": float(self.ep_p0),
            "channel_mode": self.ep_channel_mode,
            "obs_mode": self.ep_obs_mode,
            "use_aoi": bool(self.ep_use_aoi),
            "aoi": np.array([link.aoi for link in self.links], dtype=np.int32),
            "collision": False,
            "success": False,
            "episode": None,
        }
