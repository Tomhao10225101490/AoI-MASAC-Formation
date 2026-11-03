"""Geometry of the fixed-footprint density sweep. No planner is run."""
import numpy as np

from planning.dc_rrt.operational_maps import make_density_map


def _separated(scene, margin=3.0):
    start = np.asarray(scene["start"], float)
    goal = np.asarray(scene["goal"], float)
    for lo, hi in scene["obstacles"]:
        lo = np.asarray(lo, float) - margin
        hi = np.asarray(hi, float) + margin
        if np.all(start >= lo) and np.all(start <= hi):
            return False
        if np.all(goal >= lo) and np.all(goal <= hi):
            return False
    boxes = [(np.asarray(lo[:2]), np.asarray(hi[:2])) for lo, hi in scene["obstacles"]]
    for i, (a, b) in enumerate(boxes):
        for c, d in boxes[i + 1:]:
            if np.all(a < d) and np.all(c < b):
                return False
    return True


def test_density_increases_as_spacing_decreases_and_endpoints_stay_outside():
    fractions = []
    for spacing in (260, 180, 120, 80):
        scenes = [make_density_map(9300000 + i, spacing) for i in range(5)]
        assert all(_separated(scene) for scene in scenes)
        fractions.append(np.mean([scene["obstacle_volume_fraction"] for scene in scenes]))
        assert len({scene["building_count"] for scene in scenes}) == 1
    assert fractions[0] < fractions[1] < fractions[2] < fractions[3]
