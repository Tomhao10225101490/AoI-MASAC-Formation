"""Default cluttered transit used by the paper environment.

The route is a short urban-style crossing, not an empty rectangle with one
circle. Three of the legs pass through a building disk, so a straight line is
both a collision and a non-line-of-sight radio path. Three other disks move,
which is the setting used by recent formation-and-avoidance work (static
obstacles plus crossing traffic). Coordinates are meters on a 1200 m by 800 m
map.
"""

from __future__ import annotations


def default_waypoints() -> list[tuple[float, float]]:
    return [
        (280.0, 420.0),
        (520.0, 560.0),
        (800.0, 320.0),
        (1080.0, 500.0),
    ]


def default_obstacles() -> list[dict]:
    """Eight static buildings and three crossing intruders."""
    return [
        # Disks sitting on the three route legs. Radii stay clear of the
        # waypoint capture circles (see tests/test_env.py).
        {"x": 400.0, "y": 490.0, "radius": 36.0, "vx": 0.0, "vy": 0.0},
        {"x": 660.0, "y": 440.0, "radius": 34.0, "vx": 0.0, "vy": 0.0},
        {"x": 940.0, "y": 410.0, "radius": 32.0, "vx": 0.0, "vy": 0.0},
        # Off-leg clutter, so going around one building runs into another.
        {"x": 250.0, "y": 280.0, "radius": 26.0, "vx": 0.0, "vy": 0.0},
        {"x": 430.0, "y": 260.0, "radius": 28.0, "vx": 0.0, "vy": 0.0},
        {"x": 700.0, "y": 640.0, "radius": 24.0, "vx": 0.0, "vy": 0.0},
        {"x": 990.0, "y": 250.0, "radius": 30.0, "vx": 0.0, "vy": 0.0},
        {"x": 560.0, "y": 150.0, "radius": 22.0, "vx": 0.0, "vy": 0.0},
        # Crossing traffic. Speeds are a few meters per second, comparable
        # to a slow aircraft, not a teleporting dot.
        {"x": 360.0, "y": 180.0, "radius": 16.0, "vx": 0.0, "vy": 7.0},
        {"x": 860.0, "y": 680.0, "radius": 16.0, "vx": -8.0, "vy": 0.0},
        {"x": 640.0, "y": 200.0, "radius": 14.0, "vx": 6.0, "vy": 4.0},
    ]
