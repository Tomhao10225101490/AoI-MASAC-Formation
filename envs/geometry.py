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


def make_formation_offsets(n_followers: int) -> list[tuple[float, float]]:
    """V-shape slots behind the leader, in the leader body frame (meters).

    Follower 0 sits back-left, follower 1 back-right, then the next row, and
    so on. Distances grow with the row index so the aircraft do not spawn
    on top of each other.
    """
    if n_followers < 1:
        raise ValueError("n_followers must be >= 1")
    if n_followers == 1:
        return [(-60.0, 0.0)]
    offsets: list[tuple[float, float]] = []
    for i in range(n_followers):
        row = i // 2 + 1
        side = -1.0 if (i % 2 == 0) else 1.0
        offsets.append((-55.0 * row, side * 40.0 * row))
    return offsets
