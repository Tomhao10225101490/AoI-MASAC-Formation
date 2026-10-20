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
    leader_xy: tuple[float, float] | np.ndarray,
    leader_heading: float,
    offset_xy: tuple[float, float] | np.ndarray,
) -> np.ndarray:
    """World coordinate of a follower's slot given a leader pose."""
    return np.asarray(leader_xy, dtype=np.float64) + body_to_world(offset_xy, leader_heading)


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
