"""Small geometry helpers shared by the environment and the tests."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def body_to_world(offset_xy: tuple[float, float] | np.ndarray, heading: float) -> np.ndarray:
    """Rotate a body-frame offset into world axes.

    Body frame: +x is the leader's nose, +y is to the leader's left.
    """
    ox = float(offset_xy[0])
    oy = float(offset_xy[1])
    c = math.cos(heading)
    s = math.sin(heading)
    return np.array([ox * c - oy * s, ox * s + oy * c], dtype=np.float64)


def desired_slot(
    leader_xyz: tuple[float, float] | np.ndarray,
    leader_heading: float,
    offset_xyz: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """World coordinate of a follower's slot given a leader pose.

    A 2D leader and a 2D offset return a 2-vector, so older calls keep
    working. If either side carries a z, the result is a 3-vector and the
    vertical offset defaults to 0. The wedge in this repo is flat: followers
    are supposed to match the leader's altitude, and climbing off that slot
    is a formation error.
    """
    leader = np.asarray(leader_xyz, dtype=np.float64).reshape(-1)
    offset = np.asarray(offset_xyz, dtype=np.float64).reshape(-1)
    xy = leader[:2] + body_to_world(offset[:2], leader_heading)
    if leader.shape[0] == 2 and offset.shape[0] == 2:
        return xy
    z = float(leader[2] if leader.shape[0] > 2 else 0.0)
    z += float(offset[2] if offset.shape[0] > 2 else 0.0)
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def point_to_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Euclidean distance from point P to the segment AB."""
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def segment_hits_disk(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    radius: float,
) -> bool:
    """True when the segment AB intersects a disk centered at C."""
    return point_to_segment_distance(cx, cy, ax, ay, bx, by) <= radius


def segment_hits_vertical_cylinder(
    ax: float,
    ay: float,
    az: float,
    bx: float,
    by: float,
    bz: float,
    cx: float,
    cy: float,
    radius: float,
    height: float,
) -> bool:
    """True when segment AB intersects a vertical cylinder.

    The cylinder stands on the ground: horizontal disk of ``radius`` centered
    at (cx, cy), occupying heights from 0 up to ``height``. A ray that crosses
    the footprint only above the roof is not a hit. That is the difference
    between a 2D disk and a building: climbing can restore line of sight.
    """
    if radius < 0.0 or height <= 0.0:
        return False
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    ox = ax - cx
    oy = ay - cy
    a = abx * abx + aby * aby
    if a <= 1e-12:
        if math.hypot(ox, oy) > radius:
            return False
        z_lo = min(az, bz)
        z_hi = max(az, bz)
        return z_lo < height and z_hi > 0.0
    b = 2.0 * (ox * abx + oy * aby)
    c = ox * ox + oy * oy - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    enter = max(0.0, t0)
    leave = min(1.0, t1)
    if enter > leave:
        return False
    z_enter = az + enter * abz
    z_leave = az + leave * abz
    z_lo = min(z_enter, z_leave)
    z_hi = max(z_enter, z_leave)
    return z_lo < height and z_hi > 0.0


def make_formation_offsets(
    n_followers: int, spacing: float = 32.0
) -> list[tuple[float, float]]:
    """Wedge behind the leader, body frame, meters.

    Body +x is the nose, +y is left. Row 1 is one spacing behind the leader;
    each further row steps back by ``spacing`` and spreads by 0.75 spacing.
    Spacing is large relative to the 4 m airframe radius so slots are not
    already in collision.
    """
    if n_followers < 1:
        raise ValueError("n_followers must be >= 1")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    if n_followers == 1:
        return [(-spacing, 0.0)]
    offsets: list[tuple[float, float]] = []
    for i in range(n_followers):
        row = i // 2 + 1
        side = -1.0 if (i % 2 == 0) else 1.0
        offsets.append((-spacing * row, side * spacing * 0.75 * row))
    return offsets
