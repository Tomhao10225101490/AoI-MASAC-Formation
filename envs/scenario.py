"""Default 3D urban transit used by the paper environment.

The map is 1200 m by 800 m in the horizontal plane. Altitude is part of the
state: some buildings are shorter than the altitude ceiling, so a formation
can fly over them and restore line of sight; one building on the route is
taller than the ceiling, so it has to be flown around and it keeps blocking
the radio link until the horizontal footprint is clear.

Three of the legs pass through a building footprint. Crossing traffic is a
sphere at a fixed altitude and does not block radio. Coordinates are meters.
"""

from __future__ import annotations


def default_waypoints() -> list[tuple[float, float, float]]:
    return [
        (280.0, 420.0, 40.0),
        (520.0, 560.0, 80.0),
        (800.0, 320.0, 40.0),
        (1080.0, 500.0, 40.0),
    ]


CANYON_STREET = 70.0
# Fixed cap so a randomized canyon never changes the critic input size.
CANYON_MAX_OBSTACLES = 64


def sample_canyon(rng) -> dict:
    """One random street canyon. Lanes stay clear of every roof.

    Street width, roof height, block spacing and the cross-street phase
    change. The route does not: the leader still flies y = 400 from x = 120
    to x = 1120, so episodes stay comparable.
    """
    street = float(rng.uniform(66.0, 84.0))
    radius = float(rng.uniform(12.0, 14.0))
    spacing = float(rng.uniform(48.0, 56.0))
    gap_every = int(rng.integers(4, 7))
    gap_phase = int(rng.integers(0, gap_every))
    mid_y = 400.0
    obstacles = []
    x = 250.0
    index = 0
    while x <= 1000.0:
        for row in (-1.5, -0.5, 0.5, 1.5):
            if index % gap_every == gap_phase:
                continue
            obstacles.append(
                {
                    "x": x + float(rng.uniform(-2.0, 2.0)),
                    "y": mid_y + row * street + float(rng.uniform(-2.0, 2.0)),
                    "radius": radius,
                    "height": float(rng.uniform(48.0, 78.0)),
                    "blocks_radio": True,
                }
            )
        x += spacing
        index += 1
    obstacles = obstacles[:CANYON_MAX_OBSTACLES]
    return {
        "obstacles": obstacles,
        "waypoints": [(600.0, mid_y, 40.0), (1120.0, mid_y, 40.0)],
        "formation_offsets": [
            (-40.0, -street, 0.0),
            (-40.0, street, 0.0),
            (-80.0, -2 * street, 0.0),
            (-80.0, 2 * street, 0.0),
        ],
        "start_xy": (120.0, mid_y),
    }


def canyon_scenario() -> dict:
    """Parallel streets separated by rows of short buildings.

    The leader flies the middle street. Each follower holds its own street,
    so the row between them cuts the downlink for most of the run. Every
    roof is under the 120 m ceiling: a follower can always buy line of
    sight by climbing, and pays for it in altitude error against its slot.
    Rows have cross streets every few blocks, which are short LoS windows.
    """
    mid_y = 400.0
    heights = (50.0, 65.0, 55.0, 75.0, 45.0, 60.0, 70.0)
    obstacles = []
    k = 0
    for row in (-1.5, -0.5, 0.5, 1.5):
        y = mid_y + row * CANYON_STREET
        x = 260.0
        while x <= 990.0:
            if int((x - 260.0) // 48.0) % 5 != 4:
                obstacles.append(
                    {"x": x, "y": y, "radius": 14.0, "height": heights[k % len(heights)], "blocks_radio": True}
                )
                k += 1
            x += 48.0
    s = CANYON_STREET
    return {
        "obstacles": obstacles,
        "waypoints": [(600.0, mid_y, 40.0), (1120.0, mid_y, 40.0)],
        "formation_offsets": [(-40.0, -s, 0.0), (-40.0, s, 0.0), (-80.0, -2 * s, 0.0), (-80.0, 2 * s, 0.0)],
        "start_xy": (120.0, mid_y),
    }


def default_obstacles() -> list[dict]:
    """Eight static buildings and three crossing intruders."""
    return [
        # Short roof on the first leg. Cruise altitude (40 m) hits it and
        # loses line of sight; climbing past the roof clears both.
        {"x": 400.0, "y": 490.0, "radius": 36.0, "height": 50.0, "blocks_radio": True, "role": "near_roof"},
        # Taller than z_max (120 m). Cannot be overflown. The radio stays
        # blocked for as long as the footprint sits between the two aircraft.
        {"x": 660.0, "y": 440.0, "radius": 34.0, "height": 160.0, "blocks_radio": True, "role": "tall"},
        # Medium roof on the last leg. Overflyable, but not at cruise altitude.
        {"x": 940.0, "y": 410.0, "radius": 32.0, "height": 70.0, "blocks_radio": True, "role": "far_roof"},
        # Off-leg clutter, mixed heights, so going around one roof meets another.
        {"x": 250.0, "y": 280.0, "radius": 26.0, "height": 90.0, "blocks_radio": True},
        {"x": 430.0, "y": 260.0, "radius": 28.0, "height": 40.0, "blocks_radio": True},
        {"x": 700.0, "y": 640.0, "radius": 24.0, "height": 110.0, "blocks_radio": True},
        {"x": 990.0, "y": 250.0, "radius": 30.0, "height": 55.0, "blocks_radio": True},
        {"x": 560.0, "y": 150.0, "radius": 22.0, "height": 45.0, "blocks_radio": True},
        # Crossing traffic. Spheres, not radio blockers.
        {
            "x": 360.0,
            "y": 180.0,
            "z": 40.0,
            "radius": 16.0,
            "height": 0.0,
            "vx": 0.0,
            "vy": 7.0,
            "blocks_radio": False,
        },
        {
            "x": 860.0,
            "y": 680.0,
            "z": 60.0,
            "radius": 16.0,
            "height": 0.0,
            "vx": -8.0,
            "vy": 0.0,
            "blocks_radio": False,
        },
        {
            "x": 640.0,
            "y": 200.0,
            "z": 35.0,
            "radius": 14.0,
            "height": 0.0,
            "vx": 6.0,
            "vy": 4.0,
            "blocks_radio": False,
        },
    ]
