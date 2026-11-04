"""Offline geometric and simultaneous, dynamics-checked reference witnesses.

This module is not used by the actor or as a training demonstrator.  Failure
to find a witness is reported as unresolved, not as proof of infeasibility.
"""
import heapq
import itertools
import math

import numpy as np

from uav_planning.structured import segment_box_hit
from uav_planning.v4_environment import V4Mission


def _line_clear(a, b, lo, hi, world, boundary):
    a, b = np.asarray(a), np.asarray(b)
    if (np.any(a <= boundary) or np.any(b <= boundary) or
            np.any(a >= world-boundary) or np.any(b >= world-boundary)):
        return False
    return not len(lo) or not segment_box_hit(a, b, lo, hi).any()


def geometric_path(scene, agent, resolution=6., tracking_clearance=1.):
    """Conservative static voxel path and exact inflated-AABB line checks."""
    cfg = scene.config
    world = np.asarray(cfg.world, dtype=float)
    boundary = cfg.uav_radius+cfg.safety_margin+tracking_clearance
    inflation = cfg.uav_radius+cfg.safety_margin+tracking_clearance
    lo = scene.boxes[:, 0]-inflation
    hi = scene.boxes[:, 1]+inflation
    start, goal = scene.starts[agent], scene.goals[agent]
    if _line_clear(start, goal, lo, hi, world, boundary):
        return np.asarray([start, goal]), dict(status='connected_direct', resolution_m=resolution)
    dims = tuple(int(x) for x in np.floor(world/resolution))
    if min(dims) < 3 or np.prod(dims) > 500000:
        raise ValueError('Grid resolution produces an unsupported grid size')
    axes = [(np.arange(n)+.5)*resolution for n in dims]
    xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    cell_lo, cell_hi = xyz-resolution/2, xyz+resolution/2
    blocked = np.any((cell_lo < boundary) | (cell_hi > world-boundary), axis=-1)
    for low, high in zip(lo, hi):
        blocked |= np.all((cell_hi >= low) & (cell_lo <= high), axis=-1)

    def connector(point):
        base = np.floor(point/resolution).astype(int)
        candidates = []
        for off in itertools.product(range(-3, 4), repeat=3):
            idx = tuple((base+off).tolist())
            if any(idx[i] < 0 or idx[i] >= dims[i] for i in range(3)) or blocked[idx]:
                continue
            center = xyz[idx]
            if _line_clear(point, center, lo, hi, world, boundary):
                candidates.append((float(np.linalg.norm(center-point)), idx))
        return min(candidates)[1] if candidates else None

    origin, destination = connector(start), connector(goal)
    if origin is None or destination is None:
        return None, dict(status='unresolved_connector', resolution_m=resolution, grid_shape=list(dims))
    distances = {origin: 0.}
    previous = {}
    heap = [(0., 0., origin)]
    expansions = 0
    while heap:
        _, cost, cell = heapq.heappop(heap)
        if cost != distances.get(cell):
            continue
        expansions += 1
        if cell == destination:
            break
        for axis in range(3):
            for direction in (-1, 1):
                near = list(cell)
                near[axis] += direction
                if near[axis] < 0 or near[axis] >= dims[axis]:
                    continue
                near = tuple(near)
                if blocked[near]:
                    continue
                trial = cost+resolution
                if trial < distances.get(near, float('inf')):
                    distances[near] = trial
                    previous[near] = cell
                    heuristic = float(np.linalg.norm(xyz[near]-xyz[destination]))
                    heapq.heappush(heap, (trial+heuristic, trial, near))
    if destination not in distances:
        return None, dict(status='unresolved_grid', resolution_m=resolution,
                          grid_shape=list(dims), expanded=expansions)
    cells = [destination]
    while cells[-1] != origin:
        cells.append(previous[cells[-1]])
    raw = [start] + [xyz[cell] for cell in reversed(cells)] + [goal]
    # Any line shortcut must pass exact continuous inflated-AABB checks.
    result = [raw[0]]
    current = 0
    while current < len(raw)-1:
        target = len(raw)-1
        while target > current+1 and not _line_clear(raw[current], raw[target], lo, hi, world, boundary):
            target -= 1
        if not _line_clear(raw[current], raw[target], lo, hi, world, boundary):
            return None, dict(status='unresolved_exact_segment', resolution_m=resolution,
                              grid_shape=list(dims), expanded=expansions)
        result.append(raw[target])
        current = target
    return np.asarray(result), dict(status='connected_grid', resolution_m=resolution,
                                    grid_shape=list(dims), expanded=expansions)


def _actions(env, paths, indices, delays, cruise_speed):
    action = np.zeros((env.n, 3))
    for i in range(env.n):
        if env.t < delays[i]:
            desired = np.zeros(3)
        else:
            target = paths[i][indices[i]]
            delta = target-env.p[i]
            if (indices[i] < len(paths[i])-1 and np.linalg.norm(delta) < .8
                    and np.linalg.norm(env.v[i]) < 1.):
                indices[i] += 1
                target = paths[i][indices[i]]
                delta = target-env.p[i]
            distance = np.linalg.norm(delta)
            desired = delta * min(.8, cruise_speed/max(distance, 1e-12))
        raw = (desired-env.v[i])/(env.amax*env.dt)
        action[i] = raw/max(1., np.linalg.norm(raw))
    return action


def joint_reference(config, seed, resolution=6., cruise_speed=7., schedules=None):
    """Find static paths then execute all agents with declared delay candidates."""
    if config.n_agents != 3:
        raise ValueError('Current joint reference is validated for exactly three UAVs')
    if not 0 < cruise_speed <= config.vmax:
        raise ValueError('Invalid cruise speed')
    env = V4Mission(config)
    env.reset(seed)
    scene = env.scene
    paths, geometry = [], []
    for agent in range(config.n_agents):
        path, status = geometric_path(scene, agent, resolution)
        geometry.append(status)
        paths.append(path)
    if any(p is None for p in paths):
        return dict(scene=scene.to_dict(), geometry=geometry, joint_reference_success=False,
                    attempts=[], traces=[], unresolved='static geometric search did not find all three paths',
                    paths=[None if p is None else p.tolist() for p in paths])
    schedules = schedules or [(0,0,0),(0,30,60),(60,30,0),(0,60,120),(120,60,0),
                              (100,100,100),(100,130,160),(160,130,100)]
    attempts = []
    traces = []
    witness = None
    for schedule in schedules:
        if len(schedule) != config.n_agents or any(int(x) != x or x < 0 for x in schedule):
            raise ValueError('Delays must contain three nonnegative integer control ticks')
        env = V4Mission(config)
        env.reset(seed)
        positions, velocities, dynamics, actions = [env.p.copy()], [env.v.copy()], [env.dynamic.copy()], []
        indices = [1]*env.n
        info, done, trunc = None, False, False
        max_acceleration = 0.
        for _ in range(env.horizon):
            action = _actions(env, paths, indices, schedule, cruise_speed)
            previous_v = env.v.copy()
            _, _, done, trunc, info = env.step(action)
            max_acceleration = max(max_acceleration, float(np.linalg.norm((env.v-previous_v)/env.dt, axis=1).max()))
            actions.append(action.copy())
            positions.append(env.p.copy())
            velocities.append(env.v.copy())
            dynamics.append(env.dynamic.copy())
            if done or trunc:
                break
        max_speed = max(float(np.linalg.norm(v, axis=1).max()) for v in velocities)
        success = bool(info['success'] and not any(info[k] for k in
                       ('uav_collision','static_collision','dynamic_collision','boundary_collision')) and
                       env.t <= env.horizon and max_acceleration <= env.amax+1e-8 and max_speed <= env.vmax+1e-8)
        attempt = dict(delays_ticks=list(schedule), steps=env.t, time_s=env.t*env.dt,
                       success=success, final_info=info, max_speed_mps=max_speed,
                       max_acceleration_mps2=max_acceleration)
        attempts.append(attempt)
        trace = dict(positions=np.asarray(positions), velocities=np.asarray(velocities),
                     dynamic=np.asarray(dynamics), actions=np.asarray(actions))
        traces.append(trace)
        if success:
            witness = trace
            break
    return dict(scene=scene.to_dict(), geometry=geometry, joint_reference_success=witness is not None,
                attempts=attempts, traces=traces, paths=[p.tolist() for p in paths], witness=witness,
                limitations='Static grid failure is unresolved, not proof of infeasibility. Successful witness is for this simulator and seed, not a learned policy or safety guarantee.')
