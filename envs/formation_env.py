"""3D cluttered leader-follower transit with a roof-aware command link.

Gym-style API, no Gymnasium dependency:

    obs, info = env.reset(seed=None, options=None)
    obs, rewards, terminated, truncated, info = env.step(actions)

What the episode is
-------------------
A leader flies a sequence of 3D waypoints across a 1200 m by 800 m urban
field. Buildings have heights. Some are shorter than the altitude ceiling,
so the formation can climb over the roof; one building on the route is
taller than the ceiling and has to be flown around. Three spheres of
crossing traffic share the volume and do not block radio. Followers keep a
flat wedge at the leader's altitude. Wind is drawn once per episode and
pushes only the horizontal ground speed.

The leader state reaches each follower on a downlink. Success probability
falls with 3D distance and drops again when the 3D segment intersects a
building (the ray is below the roof). A lost packet keeps the previous measurement, runs one constant-velocity
predict, and increments AoI. The mean heading stays at the last packet.
Heading variance still grows, because the follower does not know the yaw
the leader applied during the outage. The proposed observation
(``obs_mode="belief"``) shows the coasted slot, the AoI, and the standard
deviation of that slot. The reward for being in formation still uses the
true leader pose, so a stale slot and the true slot disagree. The follower's
heading-shaping term uses only the heading that follower is allowed to see.

Onboard sensing of obstacles, and of the nearest *other follower*, is
limited by ``sensor_range`` and is not sent on the lossy link. A follower
never receives the leader's true position through that sensor. The leader
does see the nearest follower. The Kalman filter's non-line-of-sight test
uses the static building map, which is treated as known city geometry, not
as a packet. The actor observation does not contain a predicted-blockage
bit; that bit used to be visible to every baseline, which made the
uncertainty ablation compare two policies that both saw the same event.

Actions are outer-loop commands at ``dt``: normalized longitudinal
acceleration, yaw rate, and vertical acceleration.

``terminated`` is success or collision. ``truncated`` is the step limit.
Success means every waypoint has been captured, every follower is inside
``formation_tol`` of the slot defined by the *true* leader pose, and the
episode never collided. The last waypoint is captured only when the leader
is inside its ball and the formation condition already holds, so the route
cannot be "finished" and then succeeded somewhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

from envs.belief import LeaderBelief
from envs.channel import packet_success_probability
from envs.geometry import (
    desired_slot,
    make_formation_offsets,
    segment_hits_vertical_cylinder,
    wrap_angle,
)
from envs.scenario import default_obstacles, default_waypoints

# Static disks in old configs have no height. Treat them as columns taller
# than the altitude ceiling so a test that parks an aircraft inside the
# footprint still collides and still loses line of sight.
_LEGACY_COLUMN_HEIGHT = 200.0

OBS_MODES = ("aoi", "perfect", "hold", "extrapolate", "belief", "belief_point")


class ObsIndex:
    """Column indices of one local observation. Same length for every agent."""

    X = 0
    Y = 1
    Z = 2
    V = 3
    SIN_H = 4
    COS_H = 5
    VZ = 6
    REF_REL_X = 7
    REF_REL_Y = 8
    REF_REL_Z = 9
    REF_V = 10
    REF_SIN = 11
    REF_COS = 12
    REF_VZ = 13
    AOI = 14
    # Slot-position standard deviation, world axes, divided by std_scale.
    # Written only for obs_mode="belief". Every other mode stores zeros,
    # which is the uncertainty ablation.
    STD_X = 15
    STD_Y = 16
    STD_Z = 17
    # Kept so the rest of the vector does not shift. Always written as 0.
    # Predicted blockage used to live here and was visible to every baseline.
    PRED_NLOS = 18
    SLOT_X = 19
    SLOT_Y = 20
    SLOT_Z = 21
    DES_REL_X = 22
    DES_REL_Y = 23
    DES_REL_Z = 24
    WP_FRAC = 25
    WIND_X = 26
    WIND_Y = 27
    TEAMMATE_REL_X = 28
    TEAMMATE_REL_Y = 29
    TEAMMATE_REL_Z = 30
    TEAMMATE_DIST = 31
    TEAMMATE_PRESENT = 32
    IS_LEADER = 33
    OBS_START = 34
    # rel_x, rel_y, rel_z, vx, vy, vz, radius, height, present
    OBS_FEATS = 9

    @staticmethod
    def dim(n_nearest_obstacles: int) -> int:
        return ObsIndex.OBS_START + ObsIndex.OBS_FEATS * int(n_nearest_obstacles)


def global_state_dim(
    n_agents: int, n_followers: int, max_obstacles: int, max_waypoints: int
) -> int:
    """Privileged critic input length. Fixed once the config is fixed.

    Per aircraft: x, y, z, v, sin, cos, vz.
    Then wind (2), waypoint fraction (1), each waypoint (x, y, z),
    each obstacle slot (x, y, z, vx, vy, vz, radius, height),
    and each follower link (AoI, belief xyz, belief velocity, believed heading,
    slot std, blocked).
    """
    return (
        7 * n_agents
        + 2
        + 1
        + 3 * max_waypoints
        + 8 * max_obstacles
        + 12 * n_followers
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
    z_min: float = 15.0
    z_max: float = 120.0
    start_z: float = 40.0
    vz_max: float = 6.0
    max_climb_accel: float = 3.0
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
    obs_mode: str = "belief"
    use_aoi: bool = True
    nlos_q_scale: float = 4.0
    belief_q_pos: float = 0.25
    belief_q_vel: float = 1.0
    belief_meas_var_pos: float = 1.0
    belief_meas_var_vel: float = 0.25
    # About 1.8 degrees. A fresh packet should not already swing the slot
    # by a large fraction of formation_tol.
    belief_meas_var_heading: float = 0.001
    belief_var_cap: float = 22500.0
    std_scale: float = 80.0
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
    # Initial weights. Not a tuned paper result.
    w_progress: float = 0.25
    w_heading: float = 0.05
    w_formation: float = 0.02
    w_form_progress: float = 0.25
    # Linear formation cost stops growing past this error. Combined with
    # collision_penalty, a crash is more expensive than flying a full
    # episode while sitting on this cap. See docs/MODEL.md section 9.
    formation_error_cap: float = 80.0
    w_smooth: float = 0.02
    w_step: float = 0.01
    w_clearance: float = 0.5
    clearance_margin: float = 20.0
    formation_bonus: float = 0.15
    collision_penalty: float = 250.0
    success_bonus: float = 40.0
    wp_bonus: float = 8.0
    oob_penalty: float = 1.0
    obstacle_speed_norm: float = 10.0

    def __post_init__(self) -> None:
        self.start_xy = (float(self.start_xy[0]), float(self.start_xy[1]))
        self.waypoints = [_as_waypoint(p, self.start_z) for p in self.waypoints]
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
        if self.obs_mode not in OBS_MODES:
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
        if self.z_max <= self.z_min:
            raise ValueError("z_max must be > z_min")
        if not self.z_min <= self.start_z <= self.z_max:
            raise ValueError("start_z must lie inside [z_min, z_max]")
        if self.vz_max <= 0.0 or self.max_climb_accel <= 0.0:
            raise ValueError("vz_max and max_climb_accel must be positive")
        if self.map_width <= 0.0 or self.map_height <= 0.0:
            raise ValueError("map size must be positive")
        if self.sensor_range <= 0.0:
            raise ValueError("sensor_range must be positive")
        if self.wind_max <= 0.0:
            raise ValueError("wind_max must be positive")
        if self.std_scale <= 0.0 or self.nlos_q_scale < 1.0:
            raise ValueError("std_scale must be positive and nlos_q_scale must be >= 1")
        if self.formation_error_cap <= 0.0:
            raise ValueError("formation_error_cap must be positive")
        if self.belief_meas_var_heading < 0.0:
            raise ValueError("belief_meas_var_heading must be non-negative")


def env_config_from_dict(data: dict) -> EnvConfig:
    known = {f.name for f in fields(EnvConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise KeyError(f"Unknown env config keys: {unknown}")
    return EnvConfig(**data)


def _as_waypoint(raw, default_z: float) -> tuple[float, float, float]:
    if len(raw) == 2:
        return (float(raw[0]), float(raw[1]), float(default_z))
    if len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    raise ValueError(f"waypoint must have 2 or 3 numbers, got {raw!r}")


@dataclass
class _UAV:
    x: float
    y: float
    z: float
    v: float
    heading: float
    vz: float


@dataclass
class _Obstacle:
    x: float
    y: float
    z: float
    radius: float
    height: float
    vx: float
    vy: float
    vz: float
    blocks_radio: bool

    @property
    def moving(self) -> bool:
        return self.vx != 0.0 or self.vy != 0.0 or self.vz != 0.0

    @classmethod
    def from_dict(cls, raw: dict, default_z: float) -> "_Obstacle":
        vx = float(raw.get("vx", 0.0))
        vy = float(raw.get("vy", 0.0))
        vz = float(raw.get("vz", 0.0))
        moving = vx != 0.0 or vy != 0.0 or vz != 0.0
        if "height" in raw:
            height = float(raw["height"])
        elif moving:
            height = 0.0
        else:
            height = _LEGACY_COLUMN_HEIGHT
        if "blocks_radio" in raw:
            blocks = bool(raw["blocks_radio"])
        else:
            blocks = height > 0.0
        z = float(raw.get("z", 0.0 if height > 0.0 else default_z))
        return cls(
            x=float(raw["x"]),
            y=float(raw["y"]),
            z=z,
            radius=float(raw["radius"]),
            height=height,
            vx=vx,
            vy=vy,
            vz=vz,
            blocks_radio=blocks and height > 0.0,
        )


@dataclass
class _Link:
    """Last accepted leader measurement, plus the coasting belief."""

    meas_x: float
    meas_y: float
    meas_z: float
    meas_v: float
    meas_heading: float
    meas_vz: float
    aoi: int
    blocked: int
    pred_blocked: int
    belief: LeaderBelief


class UAVFormationEnv:
    """3D waypoint transit, wedge formation, roof-aware lossy leader link."""

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config if config is not None else EnvConfig()
        self.n_followers = int(self.cfg.n_followers)
        self.n_agents = self.n_followers + 1
        self.action_dim = 3
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
            self.offsets = [tuple(float(v) for v in p) for p in self.cfg.formation_offsets]
        raw_obstacles = list(self.cfg.obstacles) + list(self.cfg.extra_obstacles)
        self._obstacle_init = [
            _Obstacle.from_dict(o, self.cfg.start_z) for o in raw_obstacles
        ]
        self.waypoints = [_as_waypoint(p, self.cfg.start_z) for p in self.cfg.waypoints]
        self.diag = float(math.hypot(self.cfg.map_width, self.cfg.map_height))
        self.alt_scale = float(self.cfg.z_max)
        self._rng = np.random.default_rng(self.cfg.seed)
        self.uavs: list[_UAV] = []
        self.obstacles: list[_Obstacle] = []
        self.links: list[_Link] = []
        self._wp_index = 0
        self._wp_done = 0
        self._step_count = 0
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.wind_obs_x = 0.0
        self.wind_obs_y = 0.0
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
        self._sample_wind_obs()
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

        self._sample_wind_obs()
        self._integrate(actions)
        self._move_obstacles()
        oob = self._apply_bounds()
        hit_obstacle, hit_uav = self._collision_flags()
        # Progress is measured against the waypoint that was current before
        # capture. Switching the index inside the same reward would bill the
        # leader for the distance to the *next* point.
        wp_dist_before = self._waypoint_distance()
        captured = self._advance_waypoints()
        wp_dist_after = self._waypoint_distance()
        packet_ok = self._update_channel()
        errors = self._formation_errors()
        rewards = self._rewards(
            actions=actions,
            errors=errors,
            progress=self._prev_wp_dist - wp_dist_before,
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
        self._accumulate_visibility(errors)
        self._prev_wp_dist = wp_dist_after
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
                    u.z / self.alt_scale,
                    u.v / cfg.v_max,
                    math.sin(u.heading),
                    math.cos(u.heading),
                    u.vz / cfg.vz_max,
                ]
            )
        parts.extend([self.wind_x / cfg.wind_max, self.wind_y / cfg.wind_max])
        parts.append(self._wp_done / len(self.waypoints))
        for k in range(cfg.max_waypoints):
            if k < len(self.waypoints):
                wx, wy, wz = self.waypoints[k]
                parts.extend([wx / cfg.map_width, wy / cfg.map_height, wz / self.alt_scale])
            else:
                parts.extend([0.0, 0.0, 0.0])
        for k in range(cfg.max_obstacles):
            if k < len(self.obstacles):
                o = self.obstacles[k]
                oz = o.height if o.height > 0.0 else o.z
                parts.extend(
                    [
                        o.x / cfg.map_width,
                        o.y / cfg.map_height,
                        oz / self.alt_scale,
                        o.vx / cfg.obstacle_speed_norm,
                        o.vy / cfg.obstacle_speed_norm,
                        o.vz / cfg.obstacle_speed_norm,
                        o.radius / self.diag,
                        o.height / self.alt_scale,
                    ]
                )
            else:
                parts.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i, link in enumerate(self.links):
            std = self._slot_std(i)
            m = link.belief.mean
            parts.extend(
                [
                    link.aoi / cfg.aoi_cap,
                    m[0] / cfg.map_width,
                    m[1] / cfg.map_height,
                    m[2] / self.alt_scale,
                    m[3] / cfg.v_max,
                    m[4] / cfg.v_max,
                    m[5] / cfg.vz_max,
                    float(m[6] / math.pi),
                    float(std[0] / cfg.std_scale),
                    float(std[1] / cfg.std_scale),
                    float(std[2] / cfg.std_scale),
                    # Geometric blockage is always counted in nlos_fraction.
                    # The critic bit follows the ablation: when use_blockage is
                    # off, the link is not treated as blocked. Actors do not
                    # receive this bit.
                    float(link.blocked if self.ep_use_blockage else 0.0),
                ]
            )
        state = np.asarray(parts, dtype=np.float32)
        if state.shape != (self.global_dim,):
            raise RuntimeError(f"global state length {state.shape[0]} != {self.global_dim}")
        return state

    def reference_leader_state(self, follower_index: int) -> np.ndarray:
        """Leader pose written into that follower's observation.

        Layout: x, y, z, horizontal speed, heading, vertical speed.
        """
        return self._reference_pose(follower_index)

    def link_blocked(self, follower_index: int) -> bool:
        """True when a building intersects the true follower–leader segment."""
        leader = self.uavs[0]
        follower = self.uavs[follower_index + 1]
        return self._radio_blocked(
            leader.x, leader.y, leader.z, follower.x, follower.y, follower.z
        )

    def snapshot(self) -> dict:
        wp = self.waypoints[min(self._wp_index, len(self.waypoints) - 1)]
        return {
            "uav_xy": np.array([[u.x, u.y] for u in self.uavs], dtype=np.float32),
            "uav_z": np.array([u.z for u in self.uavs], dtype=np.float32),
            "heading": np.array([u.heading for u in self.uavs], dtype=np.float32),
            "obstacle_xy": np.array(
                [[o.x, o.y] for o in self.obstacles], dtype=np.float32
            ).reshape(-1, 2),
            "obstacle_r": np.array([o.radius for o in self.obstacles], dtype=np.float32),
            "obstacle_height": np.array(
                [o.height for o in self.obstacles], dtype=np.float32
            ),
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
        if obs_mode not in OBS_MODES:
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

    def _sample_wind_obs(self) -> None:
        """One noisy wind estimate per step, shared by every follower.

        Drawn here, not inside ``_observations``, so reading the observation
        twice does not change the belief predict.
        """
        cfg = self.cfg
        self.wind_obs_x = self.wind_x + float(self._rng.normal(0.0, cfg.wind_obs_noise))
        self.wind_obs_y = self.wind_y + float(self._rng.normal(0.0, cfg.wind_obs_noise))

    def _spawn(self) -> None:
        cfg = self.cfg
        leader = self._sample_leader_pose()
        self.uavs = [leader]
        for offset in self.offsets:
            slot = self._slot_from((leader.x, leader.y, leader.z), leader.heading, offset)
            placed = False
            for _ in range(40):
                noise = self._rng.uniform(-cfg.start_noise, cfg.start_noise, size=2)
                x = float(slot[0] + noise[0])
                y = float(slot[1] + noise[1])
                z = float(slot[2])
                heading = wrap_angle(leader.heading + float(self._rng.uniform(-0.05, 0.05)))
                if self._point_clear(x, y, z):
                    self.uavs.append(_UAV(x, y, z, cfg.start_speed, heading, 0.0))
                    placed = True
                    break
            if not placed:
                x = float(np.clip(slot[0], cfg.uav_radius, cfg.map_width - cfg.uav_radius))
                y = float(np.clip(slot[1], cfg.uav_radius, cfg.map_height - cfg.uav_radius))
                self.uavs.append(_UAV(x, y, float(slot[2]), cfg.start_speed, leader.heading, 0.0))
        self.obstacles = [
            _Obstacle(
                o.x, o.y, o.z, o.radius, o.height, o.vx, o.vy, o.vz, o.blocks_radio
            )
            for o in self._obstacle_init
        ]

    def _sample_leader_pose(self) -> _UAV:
        cfg = self.cfg
        for _ in range(40):
            x = float(cfg.start_xy[0] + self._rng.uniform(-cfg.start_noise, cfg.start_noise))
            y = float(cfg.start_xy[1] + self._rng.uniform(-cfg.start_noise, cfg.start_noise))
            x = float(np.clip(x, cfg.uav_radius, cfg.map_width - cfg.uav_radius))
            y = float(np.clip(y, cfg.uav_radius, cfg.map_height - cfg.uav_radius))
            if self._point_clear(x, y, cfg.start_z):
                return _UAV(x, y, cfg.start_z, cfg.start_speed, 0.0, 0.0)
        return _UAV(float(cfg.start_xy[0]), float(cfg.start_xy[1]), cfg.start_z, cfg.start_speed, 0.0, 0.0)

    def _point_clear(self, x: float, y: float, z: float) -> bool:
        cfg = self.cfg
        if x < cfg.uav_radius or y < cfg.uav_radius:
            return False
        if x > cfg.map_width - cfg.uav_radius or y > cfg.map_height - cfg.uav_radius:
            return False
        if z < cfg.z_min or z > cfg.z_max:
            return False
        probe = _UAV(x, y, z, 0.0, 0.0, 0.0)
        for o in self._obstacle_init:
            if self._gap_to_obstacle(probe, o) < 8.0:
                return False
        for u in self.uavs:
            if self._uav_distance(probe, u) < 2.5 * cfg.uav_radius:
                return False
        return True

    def _new_belief(self) -> LeaderBelief:
        cfg = self.cfg
        return LeaderBelief(
            meas_var_pos=cfg.belief_meas_var_pos,
            meas_var_vel=cfg.belief_meas_var_vel,
            meas_var_heading=cfg.belief_meas_var_heading,
            q_pos=cfg.belief_q_pos,
            q_vel=cfg.belief_q_vel,
            var_cap=cfg.belief_var_cap,
        )

    def _write_measurement(self, link: _Link, leader: _UAV) -> None:
        link.meas_x = leader.x
        link.meas_y = leader.y
        link.meas_z = leader.z
        link.meas_v = leader.v
        link.meas_heading = leader.heading
        link.meas_vz = leader.vz
        link.belief.reset_from_measurement(
            leader.x,
            leader.y,
            leader.z,
            leader.v * math.cos(leader.heading),
            leader.v * math.sin(leader.heading),
            leader.vz,
            leader.heading,
        )

    def reset_belief_from_stored_measurement(self, follower_index: int) -> None:
        """Rebuild the coast from ``meas_*``. Used by tests that edit a packet."""
        link = self.links[follower_index]
        link.belief.reset_from_measurement(
            link.meas_x,
            link.meas_y,
            link.meas_z,
            link.meas_v * math.cos(link.meas_heading),
            link.meas_v * math.sin(link.meas_heading),
            link.meas_vz,
            link.meas_heading,
        )

    def _deliver_initial_leader_state(self) -> None:
        """Free delivery at reset. Not counted in the packet-success rate."""
        leader = self.uavs[0]
        self.links = []
        for _ in range(self.n_followers):
            link = _Link(
                meas_x=leader.x,
                meas_y=leader.y,
                meas_z=leader.z,
                meas_v=leader.v,
                meas_heading=leader.heading,
                meas_vz=leader.vz,
                aoi=0,
                blocked=0,
                pred_blocked=0,
                belief=self._new_belief(),
            )
            self._write_measurement(link, leader)
            self.links.append(link)

    def _integrate(self, actions: np.ndarray) -> None:
        cfg = self.cfg
        for u, a in zip(self.uavs, actions):
            accel = float(a[0]) * cfg.max_accel
            yaw_rate = float(a[1]) * cfg.max_yaw_rate
            climb = float(a[2]) * cfg.max_climb_accel
            u.v = float(np.clip(u.v + accel * cfg.dt, cfg.v_min, cfg.v_max))
            u.heading = wrap_angle(u.heading + yaw_rate * cfg.dt)
            u.vz = float(np.clip(u.vz + climb * cfg.dt, -cfg.vz_max, cfg.vz_max))
            u.x += (u.v * math.cos(u.heading) + self.wind_x) * cfg.dt
            u.y += (u.v * math.sin(u.heading) + self.wind_y) * cfg.dt
            u.z += u.vz * cfg.dt

    def _move_obstacles(self) -> None:
        cfg = self.cfg
        for o in self.obstacles:
            if not o.moving:
                continue
            o.x += o.vx * cfg.dt
            o.y += o.vy * cfg.dt
            o.z += o.vz * cfg.dt
            if o.x < o.radius or o.x > cfg.map_width - o.radius:
                o.vx *= -1.0
                o.x = float(np.clip(o.x, o.radius, cfg.map_width - o.radius))
            if o.y < o.radius or o.y > cfg.map_height - o.radius:
                o.vy *= -1.0
                o.y = float(np.clip(o.y, o.radius, cfg.map_height - o.radius))
            if o.vz != 0.0 and (o.z < cfg.z_min or o.z > cfg.z_max):
                o.vz *= -1.0
                o.z = float(np.clip(o.z, cfg.z_min, cfg.z_max))

    def _apply_bounds(self) -> np.ndarray:
        cfg = self.cfg
        oob = np.zeros(self.n_agents, dtype=bool)
        for i, u in enumerate(self.uavs):
            cx = float(np.clip(u.x, cfg.uav_radius, cfg.map_width - cfg.uav_radius))
            cy = float(np.clip(u.y, cfg.uav_radius, cfg.map_height - cfg.uav_radius))
            cz = float(np.clip(u.z, cfg.z_min, cfg.z_max))
            if cx != u.x or cy != u.y or cz != u.z:
                oob[i] = True
                u.x, u.y, u.z = cx, cy, cz
        return oob

    def _uav_distance(self, a: _UAV, b: _UAV) -> float:
        return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))

    def _gap_to_obstacle(self, uav: _UAV, obs: _Obstacle) -> float:
        """Positive means separation, negative means penetration."""
        cfg = self.cfg
        horiz = math.hypot(uav.x - obs.x, uav.y - obs.y) - obs.radius - cfg.uav_radius
        if obs.height <= 0.0:
            return math.dist((uav.x, uav.y, uav.z), (obs.x, obs.y, obs.z)) - obs.radius - cfg.uav_radius
        # Finite building. Above the roof, horizontal overlap is not a hit.
        if horiz >= 0.0:
            return horiz
        vertical = uav.z - obs.height - cfg.uav_radius
        if uav.z >= obs.height:
            return vertical
        return horiz

    def _collision_flags(self) -> tuple[bool, bool]:
        hit_obstacle = False
        for u in self.uavs:
            for o in self.obstacles:
                gap = self._gap_to_obstacle(u, o)
                self._min_clearance = min(self._min_clearance, gap)
                if gap < 0.0:
                    hit_obstacle = True
        hit_uav = False
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self._uav_distance(self.uavs[i], self.uavs[j]) < 2.0 * self.cfg.uav_radius:
                    hit_uav = True
        return hit_obstacle, hit_uav

    def _advance_waypoints(self) -> int:
        """Capture waypoints the leader is inside.

        Intermediate points fall as soon as the leader enters the ball.
        The last point waits until every follower is inside ``formation_tol``,
        so success cannot be declared after the leader has already left.
        """
        captured = 0
        while self._wp_done < len(self.waypoints) and self._waypoint_distance() <= self.cfg.wp_radius:
            is_last = self._wp_done == len(self.waypoints) - 1
            if is_last and not bool(np.all(self._formation_errors() <= self.cfg.formation_tol)):
                break
            self._wp_done += 1
            captured += 1
            if self._wp_index < len(self.waypoints) - 1:
                self._wp_index += 1
            else:
                break
        return captured

    def _radio_blocked(
        self, ax: float, ay: float, az: float, bx: float, by: float, bz: float
    ) -> bool:
        margin = float(self.cfg.blockage_margin)
        for o in self.obstacles:
            if not o.blocks_radio:
                continue
            if segment_hits_vertical_cylinder(
                ax, ay, az, bx, by, bz, o.x, o.y, o.radius + margin, o.height
            ):
                return True
        return False

    def _update_channel(self) -> np.ndarray:
        leader = self.uavs[0]
        ok = np.zeros(self.n_followers, dtype=np.float32)
        for i, link in enumerate(self.links):
            follower = self.uavs[i + 1]
            dist = math.dist(
                (follower.x, follower.y, follower.z), (leader.x, leader.y, leader.z)
            )
            blocked = self._radio_blocked(
                leader.x, leader.y, leader.z, follower.x, follower.y, follower.z
            )
            link.blocked = int(blocked)
            if blocked:
                self._nlos_count += 1
            apply_block = bool(
                self.ep_use_blockage and self.ep_channel_mode == "distance" and blocked
            )
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
                self._write_measurement(link, leader)
                link.aoi = 0
                link.pred_blocked = int(
                    self.ep_use_blockage
                    and self._radio_blocked(
                        link.belief.mean[0],
                        link.belief.mean[1],
                        link.belief.mean[2],
                        follower.x,
                        follower.y,
                        follower.z,
                    )
                )
                ok[i] = 1.0
            else:
                self._predict_belief(link, follower)
                link.aoi = int(min(link.aoi + 1, self.cfg.aoi_cap))
                self._outage_streak[i] += 1
                ok[i] = 0.0
            std = self._slot_std(i)
            self._belief_std_sum += float(np.mean(std))
            self._aoi_sum += link.aoi
            self._aoi_max = max(self._aoi_max, link.aoi)
        self._packets_ok += int(ok.sum())
        self._packets_attempted += self.n_followers
        return ok

    def _heading_process_variance(self) -> float:
        """Variance of one coast step of heading when yaw is unknown.

        Yaw rate is treated as uniform on [-max_yaw_rate, max_yaw_rate], so
        the variance of the rate is max_yaw_rate^2 / 3 and one step of
        heading moves by that times dt^2. This is applied on every lost
        packet, not only behind a building: an outage can hide a turn in
        open air too. NLOS multiplies it further.
        """
        cfg = self.cfg
        return (cfg.max_yaw_rate * cfg.dt) ** 2 / 3.0

    def _predict_belief(self, link: _Link, follower: _UAV) -> None:
        """One coast step. NLOS against the predicted position inflates Q."""
        cfg = self.cfg
        # Mean first, so the blockage test sees where the coast lands.
        # The heading mean does not move.
        link.belief.drift(cfg.dt, self.wind_obs_x, self.wind_obs_y)
        predicted_blocked = self._radio_blocked(
            link.belief.mean[0],
            link.belief.mean[1],
            link.belief.mean[2],
            follower.x,
            follower.y,
            follower.z,
        )
        link.pred_blocked = int(self.ep_use_blockage and predicted_blocked)
        q_scale = cfg.nlos_q_scale if link.pred_blocked else 1.0
        link.belief.add_process_noise(q_scale, self._heading_process_variance())

    def _slot_std(self, follower_index: int) -> np.ndarray:
        """World-frame standard deviation of that follower's commanded slot."""
        offset = self._offset_xyz(follower_index)
        return self.links[follower_index].belief.slot_std(offset[:2])

    def _reference_pose(self, follower_index: int) -> np.ndarray:
        link = self.links[follower_index]
        leader = self.uavs[0]
        if self.ep_obs_mode == "perfect":
            return np.array(
                [leader.x, leader.y, leader.z, leader.v, leader.heading, leader.vz],
                dtype=np.float64,
            )
        if self.ep_obs_mode in ("belief", "belief_point", "extrapolate"):
            mean = link.belief.mean
            x, y, z, vx, vy, vz = (float(v) for v in mean[:6])
            speed = math.hypot(vx, vy)
            # Mean heading is the last packet. It is not the true heading.
            return np.array([x, y, z, speed, float(mean[6]), vz], dtype=np.float64)
        return np.array(
            [
                link.meas_x,
                link.meas_y,
                link.meas_z,
                link.meas_v,
                link.meas_heading,
                link.meas_vz,
            ],
            dtype=np.float64,
        )

    def _observations(self) -> list[np.ndarray]:
        cfg = self.cfg
        obs_list: list[np.ndarray] = []
        for i, u in enumerate(self.uavs):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            obs[ObsIndex.X] = u.x / cfg.map_width
            obs[ObsIndex.Y] = u.y / cfg.map_height
            obs[ObsIndex.Z] = u.z / self.alt_scale
            obs[ObsIndex.V] = u.v / cfg.v_max
            obs[ObsIndex.SIN_H] = math.sin(u.heading)
            obs[ObsIndex.COS_H] = math.cos(u.heading)
            obs[ObsIndex.VZ] = u.vz / cfg.vz_max
            obs[ObsIndex.WIND_X] = self.wind_obs_x / cfg.wind_max
            obs[ObsIndex.WIND_Y] = self.wind_obs_y / cfg.wind_max
            if i == 0:
                ref_x, ref_y, ref_z = self.waypoints[self._wp_index]
                ref_v = 0.0
                ref_h = math.atan2(ref_y - u.y, ref_x - u.x)
                ref_vz = 0.0
                aoi_norm = 0.0
                std = (0.0, 0.0, 0.0)
                pred_nlos = 0.0
                slot = (0.0, 0.0, 0.0)
                des_x, des_y, des_z = ref_x, ref_y, ref_z
                obs[ObsIndex.IS_LEADER] = 1.0
                obs[ObsIndex.WP_FRAC] = self._wp_done / len(self.waypoints)
            else:
                ref = self._reference_pose(i - 1)
                ref_x, ref_y, ref_z, ref_v, ref_h, ref_vz = (float(v) for v in ref)
                link = self.links[i - 1]
                show_aoi = self.ep_obs_mode in ("aoi", "belief", "belief_point") and self.ep_use_aoi
                aoi_norm = (link.aoi / cfg.aoi_cap) if show_aoi else 0.0
                if self.ep_obs_mode == "belief":
                    raw_std = self._slot_std(i - 1)
                    std = tuple(float(s / cfg.std_scale) for s in raw_std)
                else:
                    std = (0.0, 0.0, 0.0)
                # Always zero. See ObsIndex.PRED_NLOS.
                pred_nlos = 0.0
                slot = self._offset_xyz(i - 1)
                des = self._slot_from((ref_x, ref_y, ref_z), ref_h, slot)
                des_x, des_y, des_z = float(des[0]), float(des[1]), float(des[2])
                obs[ObsIndex.IS_LEADER] = 0.0
                obs[ObsIndex.WP_FRAC] = 0.0
            obs[ObsIndex.REF_REL_X] = (ref_x - u.x) / cfg.map_width
            obs[ObsIndex.REF_REL_Y] = (ref_y - u.y) / cfg.map_height
            obs[ObsIndex.REF_REL_Z] = (ref_z - u.z) / self.alt_scale
            obs[ObsIndex.REF_V] = ref_v / cfg.v_max
            obs[ObsIndex.REF_SIN] = math.sin(ref_h)
            obs[ObsIndex.REF_COS] = math.cos(ref_h)
            obs[ObsIndex.REF_VZ] = ref_vz / cfg.vz_max
            obs[ObsIndex.AOI] = aoi_norm
            obs[ObsIndex.STD_X] = std[0]
            obs[ObsIndex.STD_Y] = std[1]
            obs[ObsIndex.STD_Z] = std[2]
            obs[ObsIndex.PRED_NLOS] = pred_nlos
            obs[ObsIndex.SLOT_X] = slot[0] / cfg.formation_scale
            obs[ObsIndex.SLOT_Y] = slot[1] / cfg.formation_scale
            obs[ObsIndex.SLOT_Z] = slot[2] / cfg.formation_scale
            obs[ObsIndex.DES_REL_X] = (des_x - u.x) / cfg.map_width
            obs[ObsIndex.DES_REL_Y] = (des_y - u.y) / cfg.map_height
            obs[ObsIndex.DES_REL_Z] = (des_z - u.z) / self.alt_scale
            self._write_teammate(obs, i)
            self._write_obstacles(obs, u)
            obs_list.append(obs)
        return obs_list

    def _offset_xyz(self, follower_index: int) -> tuple[float, float, float]:
        raw = self.offsets[follower_index]
        if len(raw) == 2:
            return (float(raw[0]), float(raw[1]), 0.0)
        return (float(raw[0]), float(raw[1]), float(raw[2]))

    def _slot_from(self, leader_xyz, heading: float, offset) -> np.ndarray:
        return np.asarray(desired_slot(leader_xyz, heading, offset), dtype=np.float64)

    def _write_teammate(self, obs: np.ndarray, index: int) -> None:
        cfg = self.cfg
        origin = self.uavs[index]
        best = None
        best_d = float("inf")
        for j, other in enumerate(self.uavs):
            if j == index:
                continue
            # Followers learn the leader through the downlink only. Copying
            # the leader's true position into the nearest-teammate slot made
            # packet loss irrelevant for any follower whose nearest aircraft
            # was the leader, which is the front row of the wedge.
            if index != 0 and j == 0:
                continue
            d = self._uav_distance(origin, other)
            if d < best_d:
                best_d = d
                best = other
        if best is None:
            return
        if best_d <= cfg.sensor_range:
            obs[ObsIndex.TEAMMATE_REL_X] = (best.x - origin.x) / cfg.map_width
            obs[ObsIndex.TEAMMATE_REL_Y] = (best.y - origin.y) / cfg.map_height
            obs[ObsIndex.TEAMMATE_REL_Z] = (best.z - origin.z) / self.alt_scale
            obs[ObsIndex.TEAMMATE_DIST] = best_d / self.diag
            obs[ObsIndex.TEAMMATE_PRESENT] = 1.0

    def _write_obstacles(self, obs: np.ndarray, uav: _UAV) -> None:
        cfg = self.cfg
        ranked = []
        for o in self.obstacles:
            d = math.dist((o.x, o.y, o.z if o.height <= 0.0 else uav.z), (uav.x, uav.y, uav.z))
            # Buildings are sensed by horizontal range to the footprint, so a
            # roof under the aircraft is still visible.
            horiz = math.hypot(o.x - uav.x, o.y - uav.y)
            sense = horiz if o.height > 0.0 else d
            if sense <= cfg.sensor_range:
                ranked.append((sense, o))
        ranked.sort(key=lambda item: item[0])
        for k in range(cfg.n_nearest_obstacles):
            base = ObsIndex.OBS_START + k * ObsIndex.OBS_FEATS
            if k < len(ranked):
                o = ranked[k][1]
                roof_or_center = o.height if o.height > 0.0 else o.z
                obs[base + 0] = (o.x - uav.x) / cfg.map_width
                obs[base + 1] = (o.y - uav.y) / cfg.map_height
                obs[base + 2] = (roof_or_center - uav.z) / self.alt_scale
                obs[base + 3] = o.vx / cfg.obstacle_speed_norm
                obs[base + 4] = o.vy / cfg.obstacle_speed_norm
                obs[base + 5] = o.vz / cfg.obstacle_speed_norm
                obs[base + 6] = o.radius / self.diag
                obs[base + 7] = o.height / self.alt_scale
                obs[base + 8] = 1.0

    def _true_slots(self) -> np.ndarray:
        leader = self.uavs[0]
        return np.stack(
            [
                self._slot_from((leader.x, leader.y, leader.z), leader.heading, off)
                for off in self.offsets
            ],
            axis=0,
        )

    def _formation_errors(self) -> np.ndarray:
        slots = self._true_slots()
        errors = np.zeros(self.n_followers, dtype=np.float64)
        for i in range(self.n_followers):
            u = self.uavs[i + 1]
            errors[i] = math.dist((u.x, u.y, u.z), (float(slots[i, 0]), float(slots[i, 1]), float(slots[i, 2])))
        return errors

    def _waypoint_distance(self) -> float:
        leader = self.uavs[0]
        wx, wy, wz = self.waypoints[self._wp_index]
        return math.dist((leader.x, leader.y, leader.z), (wx, wy, wz))

    def _nearest_gap(self, uav: _UAV) -> float:
        if not self.obstacles:
            return float("inf")
        return min(self._gap_to_obstacle(uav, o) for o in self.obstacles)

    def _rewards(
        self,
        actions: np.ndarray,
        errors: np.ndarray,
        progress: float,
        captured: int,
        hit_obstacle: bool,
        hit_uav: bool,
        oob: np.ndarray,
    ) -> np.ndarray:
        """Per-agent dense reward. Formulas are in the README and docs/MODEL.md."""
        cfg = self.cfg
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        leader = self.uavs[0]
        wx, wy, _wz = self.waypoints[self._wp_index]
        bearing = math.atan2(wy - leader.y, wx - leader.x)
        smooth0 = float(np.sum(np.square(actions[0])))
        rewards[0] = (
            cfg.w_progress * progress
            + cfg.w_heading * math.cos(wrap_angle(bearing - leader.heading))
            - cfg.w_smooth * smooth0
            - cfg.w_step
        )
        for i in range(self.n_followers):
            smooth = float(np.sum(np.square(actions[i + 1])))
            # Match the heading this follower is allowed to know. The true
            # heading still enters the formation error below, so a stale
            # heading is paid for once, as slot error, not as a second
            # privileged heading bonus.
            known_heading = float(self._reference_pose(i)[4])
            align = math.cos(wrap_angle(known_heading - self.uavs[i + 1].heading))
            bonus = cfg.formation_bonus if errors[i] <= cfg.formation_tol else 0.0
            held_error = min(float(errors[i]), cfg.formation_error_cap)
            rewards[i + 1] = (
                cfg.w_form_progress * (self._prev_form_err[i] - errors[i])
                - cfg.w_formation * held_error
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
        self._belief_std_sum = 0.0
        self._nlos_err_sum = 0.0
        self._los_err_sum = 0.0
        self._nlos_aoi_sum = 0.0
        self._los_aoi_sum = 0.0
        self._nlos_std_sum = 0.0
        self._los_std_sum = 0.0
        self._nlos_link_steps = 0
        self._los_link_steps = 0
        self._outage_streak = np.zeros(self.n_followers, dtype=np.int32)
        self._outage_lengths: list[int] = []
        self._stat_steps = 0
        self._min_clearance = float("inf")

    def _accumulate(self, errors: np.ndarray, in_formation: bool) -> None:
        self._stat_steps += 1
        self._form_err_sum += float(errors.mean())
        if in_formation:
            self._form_keep_steps += 1

    def _accumulate_visibility(self, errors: np.ndarray) -> None:
        """Split formation error, AoI, and slot std by geometric blockage.

        One sample is one follower on one step. A step can contribute to
        both sides when only some followers are behind a roof. Episodes
        with no samples on a side report NaN for that side so a later
        average does not treat "never blocked" as zero error.
        """
        for i, link in enumerate(self.links):
            slot_std = float(np.mean(self._slot_std(i)))
            if link.blocked:
                self._nlos_err_sum += float(errors[i])
                self._nlos_aoi_sum += float(link.aoi)
                self._nlos_std_sum += slot_std
                self._nlos_link_steps += 1
            else:
                self._los_err_sum += float(errors[i])
                self._los_aoi_sum += float(link.aoi)
                self._los_std_sum += slot_std
                self._los_link_steps += 1

    @staticmethod
    def _mean_or_nan(total: float, count: int) -> float:
        if count <= 0:
            return float("nan")
        return float(total) / float(count)

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
            # Slot standard deviation in meters, not leader-position std.
            # The name is kept so existing CSV readers still find the column.
            "mean_belief_std": (
                self._belief_std_sum / (steps * self.n_followers)
                if self._packets_attempted
                else 0.0
            ),
            "los_mean_formation_error": self._mean_or_nan(self._los_err_sum, self._los_link_steps),
            "nlos_mean_formation_error": self._mean_or_nan(
                self._nlos_err_sum, self._nlos_link_steps
            ),
            "los_mean_aoi": self._mean_or_nan(self._los_aoi_sum, self._los_link_steps),
            "nlos_mean_aoi": self._mean_or_nan(self._nlos_aoi_sum, self._nlos_link_steps),
            "los_mean_slot_std": self._mean_or_nan(self._los_std_sum, self._los_link_steps),
            "nlos_mean_slot_std": self._mean_or_nan(self._nlos_std_sum, self._nlos_link_steps),
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
