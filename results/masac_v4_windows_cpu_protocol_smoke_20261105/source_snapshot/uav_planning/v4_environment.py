"""Configurable V4 mission with V3-compatible masked local observations.

Map geometry is privileged to the simulator.  No reference waypoint, future
obstacle state or full scene object is returned in the actor observation.
"""
import numpy as np

from uav_planning.environment import swept_distance
from uav_planning.structured import box_distance, limited, segment_box_hit
from uav_planning.v4_maps import SceneConfig, generate_scene


class V4Mission:
    obs_dim = 162

    def __init__(self, config=None):
        self.config = (config or SceneConfig()).validate()
        self.n = self.config.n_agents
        self.dt = self.config.dt
        self.vmax = self.config.vmax
        self.amax = self.config.amax
        self.horizon = self.config.horizon
        self.radius = self.config.uav_radius
        self.margin = self.config.safety_margin
        self.world = np.asarray(self.config.world, dtype=float)

    def reset(self, seed):
        self.scene = generate_scene(self.config, seed)
        self.seed = int(seed)
        self.p = self.scene.starts.copy()
        self.goal = self.scene.goals.copy()
        self.boxes = self.scene.boxes.copy()
        self.lo, self.hi = self.boxes[:, 0], self.boxes[:, 1]
        self.dynamic = self.scene.dynamic_positions.copy()
        self.dv = self.scene.dynamic_velocities.copy()
        self.dr = self.scene.dynamic_radii.copy()
        self.v = np.zeros_like(self.p)
        self.prev_acc = np.zeros_like(self.p)
        self.captured = np.zeros(self.n, dtype=bool)
        self.length_by_uav = np.zeros(self.n)
        self.length = 0.
        self.jerk = 0.
        self.minimum_separation = float('inf')
        self.minimum_clearance = float('inf')
        self.t = 0
        self.last_margins = np.zeros((self.n, 19))
        return self.observe()

    def metadata(self):
        return self.scene.to_dict()

    @staticmethod
    def _visible_order(distances, limit, sensor_range):
        distances = np.asarray(distances)
        ids = np.arange(len(distances))
        visible = ids[distances <= sensor_range]
        if not len(visible):
            return visible
        order = np.lexsort((visible, distances[visible]))
        return visible[order[:limit]]

    def observe(self):
        observations = []
        scale = float(max(self.world))
        for i in range(self.n):
            own = np.r_[(self.goal[i]-self.p[i])/scale, self.v[i]/self.vmax,
                         self.prev_acc[i]/self.amax, self.p[i]/self.world,
                         self.p[i]/self.world, (self.world-self.p[i])/self.world,
                         self.vmax/20, self.amax/10]
            peers = [j for j in range(self.n) if j != i]
            peer_dist = np.linalg.norm(self.p[peers]-self.p[i], axis=-1)
            selected = self._visible_order(peer_dist, 5, self.config.sensor_range)
            peer_slots = np.zeros((5, 7))
            if len(selected):
                ids = np.asarray(peers)[selected]
                peer_slots[:len(ids)] = np.c_[(self.p[ids]-self.p[i])/scale,
                                               (self.v[ids]-self.v[i])/self.vmax,
                                               np.ones(len(ids))]
            building_slots = np.zeros((8, 7))
            if len(self.boxes):
                distances = box_distance(self.p[i], self.lo, self.hi)
                ids = self._visible_order(distances, 8, self.config.sensor_range)
                if len(ids):
                    building_slots[:len(ids)] = np.c_[((self.lo[ids]+self.hi[ids])/2-self.p[i])/scale,
                                                       (self.hi[ids]-self.lo[ids])/(2*scale),
                                                       np.ones(len(ids))]
            moving_slots = np.zeros((4, 8))
            if len(self.dynamic):
                distances = np.linalg.norm(self.dynamic-self.p[i], axis=-1)-self.dr
                ids = self._visible_order(distances, 4, self.config.sensor_range)
                if len(ids):
                    moving_slots[:len(ids)] = np.c_[(self.dynamic[ids]-self.p[i])/scale,
                                                     (self.dv[ids]-self.v[i])/self.vmax,
                                                     self.dr[ids]/10, np.ones(len(ids))]
            observations.append(np.r_[own, peer_slots.ravel(), building_slots.ravel(),
                                      moving_slots.ravel(), np.zeros(19)])
        return np.asarray(observations, dtype=np.float32)

    def _advance_dynamic(self):
        if self.config.dynamic_motion == 'bounce' and len(self.dynamic):
            low, high = .05*self.world[1], .95*self.world[1]
            reverse = ((self.dynamic[:, 1] >= high) & (self.dv[:, 1] > 0)) | (
                       (self.dynamic[:, 1] <= low) & (self.dv[:, 1] < 0))
            self.dv[reverse, 1] *= -1
        self.dynamic += self.dt*self.dv

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n, 3) or not np.isfinite(action).all():
            raise ValueError('Expected finite per-UAV 3D actions')
        old = self.p.copy()
        old_dynamic = self.dynamic.copy()
        before = np.linalg.norm(self.goal-self.p, axis=-1)
        old_v = self.v.copy()
        self.v = limited(self.v, np.clip(action, -1, 1), self.dt, self.amax, self.vmax)
        acc = (self.v-old_v)/self.dt
        self.p += self.dt*self.v
        self._advance_dynamic()

        sep = swept_distance(old[:,None]-old[None], self.p[:,None]-self.p[None])
        np.fill_diagonal(sep, np.inf)
        minsep = float(sep.min())
        self.minimum_separation = min(self.minimum_separation, minsep)
        peer = bool(np.any(sep < 2*self.radius+self.margin))
        inflation = self.radius+self.margin
        static = bool(len(self.boxes) and segment_box_hit(old, self.p,
                                self.lo-inflation, self.hi+inflation).any())
        if len(self.dynamic):
            moving_distance = swept_distance(old[:,None]-old_dynamic[None],
                                            self.p[:,None]-self.dynamic[None])
            moving = bool(np.any(moving_distance < self.dr[None,:]+inflation))
            min_dynamic = float(np.min(moving_distance-self.dr[None,:]-self.radius))
        else:
            moving = False
            min_dynamic = float('inf')
        boundary = bool(np.any(self.p < inflation) or np.any(self.p > self.world-inflation))
        min_boundary = float(np.minimum(self.p-self.radius, self.world-self.radius-self.p).min())
        min_static = float(np.min(box_distance(self.p, self.lo, self.hi))-self.radius) if len(self.boxes) else float('inf')
        self.minimum_clearance = min(self.minimum_clearance, min_boundary, min_static,
                                     min_dynamic, minsep-2*self.radius)
        after = np.linalg.norm(self.goal-self.p, axis=-1)
        arrived = (after < 4) & (np.linalg.norm(self.v, axis=-1) < 2)
        delta = arrived.astype(float)-self.captured.astype(float)
        self.captured = arrived
        reward = (before-after)/2-.02-.002*np.sum(acc*acc, axis=-1)+5*delta
        collision = peer or static or moving or boundary
        if collision:
            reward -= 20
        success = bool(arrived.all() and not collision)
        if success:
            reward += 20
        self.length_by_uav += np.linalg.norm(self.p-old, axis=-1)
        self.length = float(self.length_by_uav.sum())
        self.jerk += float(np.sum(((acc-self.prev_acc)/self.dt)**2)*self.dt)
        self.prev_acc = acc
        self.t += 1
        info = dict(success=success, uav_collision=peer, static_collision=static,
                    dynamic_collision=moving, boundary_collision=boundary,
                    arrived_fraction=float(arrived.mean()), arrived_by_uav=arrived.tolist(),
                    path_length=self.length, path_length_by_uav=self.length_by_uav.tolist(),
                    time=self.t*self.dt, min_separation=self.minimum_separation,
                    min_clearance=self.minimum_clearance, jerk_integral=self.jerk,
                    remaining_distance=float(after.mean()), scene_sha256=self.scene.scene_hash)
        return self.observe(), float(np.mean(reward)), bool(collision or success), self.t >= self.horizon, info
