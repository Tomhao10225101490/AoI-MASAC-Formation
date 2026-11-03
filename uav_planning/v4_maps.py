"""Versioned, seeded procedural scenes for V4's independent-goal missions.

Geometry is privileged simulator state.  Actors receive only the masked local
observation built by v4_environment, never this scene or reference routes.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from uav_planning.structured import segment_box_hit


FAMILIES = ('urban_canyon', 'warehouse', 'courtyard', 'overpass',
            'narrow_crossing', 'dynamic_crossing')
LEVELS = ('L0', 'L1', 'L2', 'L3')


@dataclass(frozen=True)
class SceneConfig:
    family: str = 'urban_canyon'
    level: str = 'L0'
    world: tuple = (240., 240., 90.)
    n_agents: int = 3
    density: float = 1.
    building_height_range: tuple = (25., 70.)
    corridor_width: float = 20.
    turn_orientation: int = 1
    goal_distance: float = 40.
    goal_heading_deg: float = 0.
    altitude_difference: float = 0.
    dynamic_count: int = 0
    dynamic_speed: float = 3.
    dynamic_motion: str = 'bounce'
    dynamic_radius: float = 2.5
    uav_radius: float = 1.
    safety_margin: float = 0.
    conflict_degree: float = 0.
    sensor_range: float = 400.
    vmax: float = 10.
    amax: float = 4.
    horizon: int = 350
    dt: float = .2

    def validate(self):
        world = np.asarray(self.world, dtype=float)
        heights = np.asarray(self.building_height_range, dtype=float)
        numeric = [*world, *heights, self.density, self.corridor_width, self.goal_distance,
                   self.goal_heading_deg, self.altitude_difference, self.dynamic_speed,
                   self.dynamic_radius, self.uav_radius, self.safety_margin,
                   self.conflict_degree, self.sensor_range, self.vmax, self.amax, self.dt]
        if world.shape != (3,) or heights.shape != (2,) or not np.isfinite(numeric).all():
            raise ValueError('World, heights and all numeric configuration values must be finite')
        if self.family not in FAMILIES or self.level not in LEVELS:
            raise ValueError('Unknown scene family or level')
        if not 3 <= self.n_agents <= 6 or any(x <= 20 for x in world):
            raise ValueError('World is too small or the agent count is outside 3–6')
        if not 0 <= self.density <= 1 or not 0 <= self.conflict_degree <= 1:
            raise ValueError('Density and conflict_degree must be in [0, 1]')
        if not 0 < heights[0] <= heights[1] <= world[2] - 2:
            raise ValueError('Building heights must fit within world height')
        if self.turn_orientation not in (-1, 1):
            raise ValueError('turn_orientation must be -1 or 1')
        if self.dynamic_motion not in ('bounce', 'linear', 'stationary'):
            raise ValueError('dynamic_motion must be bounce, linear or stationary')
        if not 0 <= self.dynamic_count <= 16 or self.dynamic_speed < 0:
            raise ValueError('Dynamic count or speed is invalid')
        if self.level != 'L3' and self.dynamic_count:
            raise ValueError('L0–L2 standard definitions use static-only scenes')
        if (self.goal_distance <= abs(self.altitude_difference) or self.corridor_width <= 0 or
                self.dynamic_radius <= 0 or self.uav_radius <= 0 or self.safety_margin < 0 or
                self.sensor_range <= 0 or self.vmax <= 0 or self.amax <= 0 or
                self.horizon <= 0 or self.dt <= 0):
            raise ValueError('Task, clearance, sensor or motion values are invalid')
        if self.level == 'L0' and self.sensor_range < float(np.linalg.norm(world)):
            raise ValueError('L0 sensor range must cover the complete world diagonal')
        if self.level == 'L0' and self.conflict_degree:
            raise ValueError('L0 uses low-conflict parallel tasks')
        return self


def default_config(family='urban_canyon', level='L0', **overrides):
    """Difficulty presets; overrides are explicit and validated."""
    defaults = {
        'L0': dict(goal_distance=40., altitude_difference=0., corridor_width=24., dynamic_count=0,
                   conflict_degree=0., sensor_range=400.),
        'L1': dict(goal_distance=150., altitude_difference=0., corridor_width=18., dynamic_count=0,
                   conflict_degree=0., sensor_range=80.),
        'L2': dict(goal_distance=150., altitude_difference=16., corridor_width=12., dynamic_count=0,
                   conflict_degree=0., sensor_range=80.),
        'L3': dict(goal_distance=150., altitude_difference=8., corridor_width=10., dynamic_count=4,
                   conflict_degree=.7, sensor_range=80.),
    }
    if level not in defaults:
        raise ValueError('Unknown scene difficulty')
    values = dict(defaults[level], family=family, level=level)
    values.update(overrides)
    # Preserve the no-cropping L0 comparison even if users enlarge the world.
    if level == 'L0' and 'sensor_range' not in overrides:
        values['sensor_range'] = float(np.linalg.norm(values.get('world', (240., 240., 90.)))) + 1.
    return SceneConfig(**values).validate()


@dataclass
class Scene:
    config: SceneConfig
    seed: int
    boxes: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    dynamic_positions: np.ndarray
    dynamic_velocities: np.ndarray
    dynamic_radii: np.ndarray
    object_ids: list
    scene_hash: str = ''

    def payload(self):
        return dict(version='masac-v4-scene-v1', config=asdict(self.config), seed=self.seed,
                    boxes=self.boxes.tolist(), starts=self.starts.tolist(), goals=self.goals.tolist(),
                    dynamic_positions=self.dynamic_positions.tolist(),
                    dynamic_velocities=self.dynamic_velocities.tolist(),
                    dynamic_radii=self.dynamic_radii.tolist(), object_ids=self.object_ids)

    def to_dict(self):
        return {**self.payload(), 'scene_sha256': self.scene_hash}


def canonical_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _box(lo, hi):
    return [list(map(float, lo)), list(map(float, hi))]


def _geometry(cfg, rng):
    w, h, z = map(float, cfg.world)
    if cfg.level == 'L0':
        # All L0 structures stay behind the default short-goal corridor.
        # Family geometry remains visibly distinct without requiring a turn.
        if cfg.density == 0:
            return [], []
        height = float(rng.uniform(*cfg.building_height_range))
        if cfg.family == 'overpass':
            deck_z = min(z-8, max(8., sum(cfg.building_height_range)/2))
            boxes = [_box((.65*w,.18*h,deck_z),(.88*w,.82*h,deck_z+4))]
            ids = ['bridge-deck']
            for i, (x,y) in enumerate(((.67*w,.22*h),(.86*w,.78*h))):
                boxes.append(_box((x-.015*w,y-.015*h,0),(x+.015*w,y+.015*h,deck_z)))
                ids.append(f'bridge-pier-{i}')
        elif cfg.family == 'courtyard':
            x0, x1, y0, y1 = .65*w, .87*w, .30*h, .70*h
            back = x0 if cfg.turn_orientation == 1 else x1-.035*w
            boxes = [_box((back,y0,0),(back+.035*w,y1,height)),
                     _box((x0,y0,0),(x1,y0+.035*h,height)),
                     _box((x0,y1-.035*h,0),(x1,y1,height))]
            ids = ['courtyard-back','courtyard-south','courtyard-north']
        elif cfg.family == 'narrow_crossing':
            gap = min(cfg.corridor_width/2, .35*h)
            boxes = [_box((.65*w,.08*h,0),(.85*w,h/2-gap,height)),
                     _box((.65*w,h/2+gap,0),(.85*w,.92*h,height))]
            ids = ['crossing-south','crossing-north']
        elif cfg.family == 'warehouse':
            boxes = [_box((.65*w,y-.055*h,0),(.87*w,y+.055*h,height))
                     for y in (.26*h,.50*h,.74*h)]
            ids = [f'warehouse-{i}' for i in range(len(boxes))]
        else:
            centers = [(0.70*w, .22*h), (.72*w, .50*h), (.70*w, .78*h)]
            boxes = [_box((x-.05*w,y-.045*h,0),(x+.05*w,y+.045*h,
                           rng.uniform(*cfg.building_height_range))) for x,y in centers]
            ids = [f'{cfg.family}-{i}' for i in range(len(boxes))]
        if cfg.density < 1:
            keep = [i for i in range(len(boxes)) if rng.random() <= cfg.density]
            boxes, ids = [boxes[i] for i in keep], [ids[i] for i in keep]
        return boxes, ids
    low, high = cfg.building_height_range
    boxes, ids = [], []
    def add(name, lo, hi):
        boxes.append(_box(lo, hi)); ids.append(name)
    if cfg.family == 'urban_canyon':
        if cfg.corridor_width > min(.15*w, .25*h)-4:
            raise ValueError('Corridor width exceeds the urban block spacing')
        for i, x in enumerate((.35*w, .5*w, .65*w)):
            for j, y in enumerate((.25*h, .5*h, .75*h)):
                if rng.random() > cfg.density:
                    continue
                half = max(2., min(.065*w, (.15*w-cfg.corridor_width)/2))
                hy = max(2., min(.07*h, (.25*h-cfg.corridor_width)/2))
                add(f'urban-{i}-{j}', (x-half, y-hy, 0),
                    (x+half, y+hy, rng.uniform(low, high)))
    elif cfg.family == 'warehouse':
        if cfg.corridor_width > .25*h-6:
            raise ValueError('Corridor width exceeds warehouse spacing')
        for i, y in enumerate((.25*h, .5*h, .75*h)):
            if rng.random() > cfg.density:
                continue
            hh = max(3., min(.075*h, (.25*h-cfg.corridor_width)/2))
            add(f'warehouse-{i}', (.34*w, y-hh, 0), (.66*w, y+hh, rng.uniform(low, high)))
    elif cfg.family == 'courtyard':
        # Three boxes form an open U; orientation selects its opening.
        x0, x1, y0, y1, thick = .36*w, .64*w, .30*h, .70*h, .04*w
        side_x = x0 if cfg.turn_orientation == 1 else x1-thick
        height = rng.uniform(low, high)
        if cfg.density > 0:
            add('courtyard-back', (side_x, y0, 0), (side_x+thick, y1, height))
            add('courtyard-south', (x0, y0, 0), (x1, y0+.04*h, height))
            add('courtyard-north', (x0, y1-.04*h, 0), (x1, y1, height))
    elif cfg.family == 'overpass':
        if cfg.density > 0:
            deck_z = max(8., min(sum(cfg.building_height_range)/2, z-8.))
            add('bridge-deck', (.30*w, .14*h, deck_z), (.70*w, .86*h, deck_z+4.))
            for i, x in enumerate((.32*w, .66*w)):
                for j, y in enumerate((.18*h, .82*h)):
                    add(f'bridge-pier-{i}-{j}', (x-.015*w, y-.015*h, 0),
                        (x+.015*w, y+.015*h, deck_z))
    elif cfg.family == 'narrow_crossing':
        if cfg.corridor_width >= .8*h:
            raise ValueError('Corridor too wide for the crossing geometry')
        mid, gap = h/2, cfg.corridor_width/2
        height = min(z-2, max(low, .75*z))
        if cfg.density > 0:
            add('crossing-south', (.36*w, .05*h, 0), (.64*w, mid-gap, height))
            add('crossing-north', (.36*w, mid+gap, 0), (.64*w, .95*h, height))
    else:
        for i, (x, y) in enumerate(((.38*w,.22*h),(.62*w,.78*h),(.62*w,.22*h),(.38*w,.78*h))):
            if rng.random() <= cfg.density:
                add(f'dynamic-background-{i}', (x-.055*w,y-.05*h,0),
                    (x+.055*w,y+.05*h,rng.uniform(low,high)))
    return boxes, ids


def generate_scene(config, seed):
    config.validate()
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Scene seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    world = np.asarray(config.world, dtype=float)
    w, h, z = world
    boxes_list, ids = _geometry(config, rng)
    boxes = np.asarray(boxes_list, dtype=float).reshape(-1, 2, 3)
    n = config.n_agents
    # Higher conflict concentrates lanes and sends alternate vehicles in the
    # opposite direction. Their endpoints remain distinct and are validated.
    spacing = (((1-config.conflict_degree)**3)*(.70*h/(n-1))
               + config.conflict_degree*1.6)
    ys = h/2 + (np.arange(n)-(n-1)/2)*spacing
    left = np.ones(n, dtype=bool) if config.conflict_degree == 0 else np.arange(n)%2 == 0
    start_x = .20*w if config.level == 'L0' else .12*w
    z0 = np.full(n, .32*z)
    starts = np.c_[np.where(left, start_x, w-start_x), ys, z0]
    if config.level == 'L0':
        # Distinct train/development map seeds must also produce distinct
        # navigation tasks, not only background building heights.
        starts += rng.uniform([-4., -4., -3.], [4., 4., 3.], size=(n,3))
    horizontal = math.sqrt(config.goal_distance**2-config.altitude_difference**2)
    angle = math.radians(config.goal_heading_deg)
    signs = np.where(left, 1., -1.)
    goals = starts + np.c_[signs*horizontal*math.cos(angle),
                           signs*horizontal*math.sin(angle),
                           np.full(n, config.altitude_difference)]
    dynamic = np.empty((config.dynamic_count, 3), dtype=float)
    velocity = np.zeros_like(dynamic)
    for i in range(config.dynamic_count):
        dynamic[i] = [.5*w, (0.2+0.6*(i+1)/(config.dynamic_count+1))*h,
                      (.20+.6*(i%3)/2)*z]
        if config.dynamic_motion != 'stationary':
            velocity[i, 1] = config.dynamic_speed * (1 if i % 2 == 0 else -1)
    if config.dynamic_count and config.family == 'dynamic_crossing':
        # A moving obstacle shares the UAV cruise altitude and is timed to
        # intersect the center lane near a nominal 7 m/s crossing.  This is a
        # reproducible traffic setup, not an oracle exposed to the actor.
        travel_t = abs(.5*w-start_x)/min(7., config.vmax) + min(7., config.vmax)/(2*config.amax)
        center_progress = min(1., max(0., abs(.5*w-start_x)/max(horizontal, 1e-12)))
        dynamic[0, 2] = starts[0, 2] + center_progress*config.altitude_difference
        dynamic[0, 1] = np.clip(.5*h-velocity[0, 1]*travel_t, .10*h, .90*h)
    radii = np.full(config.dynamic_count, config.dynamic_radius, dtype=float)
    clearance = config.uav_radius + config.safety_margin
    for label, positions in [('starts', starts), ('goals', goals)]:
        if np.any(positions <= clearance) or np.any(positions >= world-clearance):
            raise ValueError(f'{label} outside the allowed boundary')
        if len(boxes) and segment_box_hit(positions, positions, boxes[:,0]-clearance,
                                         boxes[:,1]+clearance).any():
            raise ValueError(f'{label} inside inflated static geometry')
        pair = np.linalg.norm(positions[:,None]-positions[None,:], axis=-1)
        np.fill_diagonal(pair, np.inf)
        if np.any(pair < 2*config.uav_radius+config.safety_margin):
            raise ValueError(f'{label} have overlapping UAVs')
        if len(dynamic) and np.any(np.linalg.norm(positions[:,None]-dynamic[None,:], axis=-1)
                                   < clearance+radii[None,:]):
            raise ValueError(f'{label} overlap a dynamic obstacle at time zero')
    if config.level == 'L0' and (len(boxes) > 8 or len(dynamic) > 4):
        raise ValueError('L0 objects exceed the no-cropping observation budget')
    scene = Scene(config, int(seed), boxes, starts, goals, dynamic, velocity, radii, ids)
    scene.scene_hash = canonical_hash(scene.payload())
    return scene
