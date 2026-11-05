"""Generic source-contract tests; no registered-case outcomes or model access."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts import audit_joint_recourse_geometry as geometry
from uav_planning.environment import swept_distance
from uav_planning.structured import DIRECTIONS, limited


def configuration():
    return dict(world=[240., 240., 90.], dt=.2, amax=4., vmax=10., body_radius=1.,
                safety_margin=0., length_scale=1., sensor_range=400.,
                starts=[[120., 120., 45.], [80., 80., 45.], [160., 160., 45.]],
                goals=[[160., 120., 45.], [40., 80., 45.], [200., 160., 45.]],
                grid_resolution=5, chunk_size=13)


def generic_case():
    return dict(id="unit_fixture_not_registered_geometry", dynamic_positions=[[100., 110., 40.], [130., 120., 50.]],
                dynamic_velocities=[[1., .5, -.25], [-.5, 0., .1]], dynamic_radii=[1., 1.5])


class GeometrySourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_limiter_matches_actual_source_with_clipping_and_active_speed_limit(self):
        cfg = configuration()
        rng = np.random.default_rng(65401)
        velocity = rng.normal(size=(47, 3))
        velocity = velocity / np.linalg.norm(velocity, axis=-1, keepdims=True) * 9.99
        action = rng.uniform(-2, 2, size=(47, 3))
        expected = limited(velocity, np.clip(action, -1, 1), cfg["dt"], cfg["amax"], cfg["vmax"])
        actual = geometry.torch_limited(torch.tensor(velocity), torch.tensor(action), cfg).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        self.assertTrue(np.any(np.linalg.norm(velocity + cfg["dt"] * cfg["amax"] *
                        np.clip(action, -1, 1) / np.maximum(1, np.linalg.norm(np.clip(action, -1, 1), axis=-1, keepdims=True)), axis=-1) > cfg["vmax"]))
        self.assertTrue(np.all(np.linalg.norm(actual, axis=-1) <= cfg["vmax"]))

    def test_swept_matches_source_interior_and_degenerate_segments(self):
        rng = np.random.default_rng(65402)
        start, end = rng.normal(size=(2, 11, 4, 3))
        end[0] = start[0]
        start[1, 0], end[1, 0] = [-3., 1., 0.], [3., 1., 0.]
        actual = geometry.torch_swept(torch.tensor(start), torch.tensor(end)).numpy()
        np.testing.assert_allclose(actual, swept_distance(start, end), rtol=0, atol=1e-12)
        self.assertAlmostEqual(actual[1, 0], 1.)

    def test_all_constraints_match_independent_numpy_source_oracle(self):
        cfg, case = configuration(), generic_case()
        rng = np.random.default_rng(65403)
        actions = rng.uniform(-1, 1, (37, 3))
        first = np.array([.2, -.3, .4])
        actual = geometry.margin_batch(cfg, case, first, torch.tensor(actions)).numpy()
        expected = np.stack([geometry.numpy_constraints(cfg, case, first, action) for action in actions])
        self.assertEqual(actual.shape, (37, 10))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    def test_source_observation_layout_validity_and_relative_velocity(self):
        cfg, case = configuration(), generic_case()
        obs = geometry.observations(cfg, case)
        self.assertEqual(obs.shape, (3, 162))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_allclose(obs[0, 9:12], np.asarray(cfg["starts"][0]) / cfg["world"])
        self.assertEqual(int((obs[0, 20:55].reshape(5, 7)[:, 6] > .5).sum()), 2)
        moving = obs[0, 111:143].reshape(4, 8)
        self.assertEqual(int((moving[:, 7] > .5).sum()), 2)
        # Source sorting uses surface distance; locate the known radius1.5slot.
        index = int(np.argmin(np.abs(moving[:, 6] - .15)))
        np.testing.assert_allclose(moving[index, 3:6] * cfg["vmax"], case["dynamic_velocities"][1], atol=1e-7)
        self.assertTrue(np.all(obs[:, 55:111] == 0))

    def test_cartesian_grid_chunk_coverage_and_effective_library_inclusion(self):
        grid = torch.cat(list(geometry.grid_chunks(5, 13, "cpu"))).numpy()
        expected = np.asarray(list(itertools.product(np.linspace(-1, 1, 5), repeat=3)))
        np.testing.assert_array_equal(grid, expected)
        projected = grid / np.maximum(1, np.linalg.norm(grid, axis=-1, keepdims=True))
        distances = np.linalg.norm(DIRECTIONS[:, None] - projected[None], axis=-1)
        self.assertLess(float(distances.min(-1).max()), 1e-12)
        rng = np.random.default_rng(65404)
        targets = rng.normal(size=(53, 3))
        targets = targets / np.maximum(1, np.linalg.norm(targets, axis=-1, keepdims=True))
        distances = np.linalg.norm(targets[:, None] - projected[None], axis=-1)
        self.assertLessEqual(float(distances.min(-1).max()), np.sqrt(3) / 4)

    def test_chunked_optimum_matches_independent_numpy_enumeration(self):
        cfg, case = configuration(), generic_case()
        first = [.2, -.3, .4]
        points = np.asarray(list(itertools.product(np.linspace(-1, 1, 5), repeat=3)))
        oracle = np.stack([geometry.numpy_constraints(cfg, case, first, action) for action in points])
        result = geometry.optimize_grid(cfg, case, first, "cpu")
        self.assertEqual(result["actions_evaluated"], 125)
        self.assertAlmostEqual(result["joint"], oracle.min(-1).max(), places=10)
        np.testing.assert_allclose(result["per_constraint_max"], oracle.max(0), rtol=0, atol=1e-10)
        self.assertLessEqual(result["joint"], result["pair"] + 1e-12)

    def test_new_artifact_writer_refuses_overwrite_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory(prefix="masac_geometry_write_") as directory:
            path = Path(directory) / "manifest.json"
            geometry.write_new(path, {"record": 1})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                geometry.write_new(path, {"record": 2})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(original), {"record": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
