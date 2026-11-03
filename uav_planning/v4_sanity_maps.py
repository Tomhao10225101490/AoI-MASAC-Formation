"""Empty / single-obstacle sanity map builders for SharedMASAC learnability.

Does NOT modify frozen uav_planning/v4_maps.py (L0 continue source hashes).
Empty maps reuse density=0 on an existing L0 family. Single-obstacle maps start
from the same empty scene then inject one AABB that does not invalidate
starts/goals, so short-horizon navigation remains feasible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from uav_planning.structured import segment_box_hit
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import default_config

SANITY_KINDS = ('empty', 'single_obstacle')
FIXTURE_DIR = Path(__file__).resolve().parents[1] / 'data' / 'masac_v4_sanity'


def sanity_config(kind='empty', sensor_range=400., **overrides):
    if kind not in SANITY_KINDS:
        raise ValueError(f'Unknown sanity kind {kind!r}; expected one of {SANITY_KINDS}')
    # Always density=0 so frozen L0 geometry stays behind the short corridor;
    # single_obstacle injects its own mid-path box after reset.
    values = dict(density=0., sensor_range=float(sensor_range))
    values.update(overrides)
    return default_config('urban_canyon', 'L0', **values)


def _single_box_for_scene(starts, goals, world, seed):
    """Place one box between start and goal of the middle UAV, offset in y.

    The box is sized to require a small lateral dodge but leave a clear path
    around it for all three short L0 goals (reference-feasible by construction).
    """
    starts = np.asarray(starts, dtype=float)
    goals = np.asarray(goals, dtype=float)
    world = np.asarray(world, dtype=float)
    rng = np.random.default_rng(int(seed) + 917_331)
    mid = 0.5 * (starts[1] + goals[1])
    half_x = 6.0
    half_y = 8.0
    height = 35.0
    # Offset slightly toward +y so the straight-line middle path clips it
    # while side lanes remain mostly clear.
    center = mid + np.array([0.0, 4.0, 0.0])
    lo = np.array([center[0] - half_x, center[1] - half_y, 0.0])
    hi = np.array([center[0] + half_x, center[1] + half_y, height])
    margin = 4.0
    lo = np.maximum(lo, margin)
    hi = np.minimum(hi, world - margin)
    if np.any(hi <= lo + 1.0):
        lo = np.array([0.30 * world[0], 0.46 * world[1], 0.0])
        hi = np.array([0.38 * world[0], 0.54 * world[1], height])
    jitter = rng.uniform(-0.4, 0.4, size=2)
    lo[:2] += jitter
    hi[:2] += jitter
    return lo, hi


def _validate_endpoints(starts, goals, boxes, world, uav_radius=1.0, safety_margin=0.0):
    clearance = uav_radius + safety_margin
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, 3)
    for label, positions in (('starts', starts), ('goals', goals)):
        positions = np.asarray(positions, dtype=float)
        if np.any(positions <= clearance) or np.any(positions >= world - clearance):
            raise ValueError(f'{label} outside the allowed boundary')
        if len(boxes) and segment_box_hit(
            positions, positions, boxes[:, 0] - clearance, boxes[:, 1] + clearance
        ).any():
            raise ValueError(f'{label} inside inflated static geometry')


def inject_single_obstacle(env, seed):
    """Mutate a density=0 V4Mission in-place with one mid-path box."""
    lo, hi = _single_box_for_scene(env.p, env.goal, env.world, seed)
    boxes = np.asarray([[lo, hi]], dtype=float)
    _validate_endpoints(env.p, env.goal, boxes, env.world, env.radius, env.margin)
    env.boxes = boxes
    env.lo, env.hi = boxes[:, 0], boxes[:, 1]
    env.scene.boxes = boxes.copy()
    env.scene.object_ids = ['sanity-single-obstacle']
    payload = env.scene.payload()
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    env.scene.scene_hash = hashlib.sha256(raw).hexdigest()
    return env


def straight_line_clear(env):
    """True if every UAV start→goal segment clears inflated boxes."""
    if not len(env.boxes):
        return True
    inflation = env.radius + env.margin
    hits = segment_box_hit(env.p, env.goal, env.lo - inflation, env.hi + inflation)
    return not bool(hits.any())


def make_sanity_env(kind, seed, sensor_range=400., **overrides):
    """Build a reset V4Mission for empty or single-obstacle sanity."""
    cfg = sanity_config(kind, sensor_range=sensor_range, **overrides)
    env = V4Mission(cfg)
    obs = env.reset(int(seed))
    if kind == 'single_obstacle':
        inject_single_obstacle(env, int(seed))
        obs = env.observe()
    return env, obs


def development_map_entries(protocol_maps):
    """Normalize protocol development_maps list to (seed, kind) entries."""
    out = []
    for entry in protocol_maps:
        kind = entry.get('kind') or entry.get('family')
        if kind not in SANITY_KINDS:
            raise ValueError(f'development map kind must be in {SANITY_KINDS}, got {kind!r}')
        out.append({'seed': int(entry['seed']), 'kind': kind, 'level': entry.get('level', 'L0')})
    return out


def write_fixture_catalog(path=None):
    """Materialize a small fixed map catalog under data/masac_v4_sanity/."""
    path = Path(path) if path is not None else FIXTURE_DIR
    path.mkdir(parents=True, exist_ok=True)
    catalog = {
        'version': 'masac-v4-sanity-maps-v1',
        'kinds': list(SANITY_KINDS),
        'maps': [],
    }
    for seed in range(870001, 870005):
        env, _ = make_sanity_env('empty', seed)
        catalog['maps'].append({
            'seed': seed,
            'kind': 'empty',
            'level': 'L0',
            'scene_sha256': env.scene.scene_hash,
            'n_boxes': int(len(env.boxes)),
            'starts': env.p.tolist(),
            'goals': env.goal.tolist(),
            'straight_line_clear': True,
        })
    for seed in range(870101, 870105):
        env, _ = make_sanity_env('single_obstacle', seed)
        catalog['maps'].append({
            'seed': seed,
            'kind': 'single_obstacle',
            'level': 'L0',
            'scene_sha256': env.scene.scene_hash,
            'n_boxes': int(len(env.boxes)),
            'boxes': env.boxes.tolist(),
            'starts': env.p.tolist(),
            'goals': env.goal.tolist(),
            'straight_line_clear': bool(straight_line_clear(env)),
            'note': 'May require a short lateral dodge; starts/goals remain outside the box.',
        })
    out = path / 'catalog.json'
    out.write_text(json.dumps(catalog, indent=2, allow_nan=False) + '\n')
    return out, catalog


__all__ = [
    'SANITY_KINDS',
    'FIXTURE_DIR',
    'sanity_config',
    'make_sanity_env',
    'inject_single_obstacle',
    'straight_line_clear',
    'development_map_entries',
    'write_fixture_catalog',
]
