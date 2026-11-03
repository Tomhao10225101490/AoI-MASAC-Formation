import json
from pathlib import Path

import numpy as np

from uav_planning.masac_v3_curriculum import make_episode
from uav_planning.structured import StructuredMission, box_distance


def config():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/masac_v3/protocol.json").read_text())
    cfg["stages"] = [{"name": "short_edge", "start_step": 0},
                     {"name": "full", "start_step": 30000}]
    return cfg


def test_curriculum_goals_are_clear_and_observation_shapes_match():
    cfg = config()
    for seed in (411, 422, 433):
        for episode in range(6):
            env, obs, stage = make_episode(seed, episode, 29999, cfg)
            assert stage == "short_edge"
            assert obs.shape == (3, 162)
            assert np.isfinite(obs).all()
            np.testing.assert_array_equal(env.goal[:, (0, 2)], env.p[:, (0, 2)])
            np.testing.assert_allclose(np.linalg.norm(env.goal-env.p, axis=1), 40.0)
            assert np.all((env.p > env.radius) & (env.p < env.world - env.radius))
            assert np.all((env.goal > env.radius) & (env.goal < env.world - env.radius))
            assert np.min(box_distance(env.p, env.lo, env.hi)) > env.radius
            assert np.min(box_distance(env.goal, env.lo, env.hi)) > env.radius
            for positions in (env.p, env.goal):
                peer_dist = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
                np.fill_diagonal(peer_dist, np.inf)
                assert peer_dist.min() > 2 * env.radius
                dynamic_dist = np.linalg.norm(positions[:, None] - env.dynamic[None, :], axis=-1)
                assert np.all(dynamic_dist > env.dr[None, :] + env.radius)
            assert env.dynamic.shape == (4, 3) and len(env.boxes) >= 8


def test_full_stage_reproduces_original_reset_and_transition():
    cfg = config()
    for episode in range(3):
        env, obs, stage = make_episode(411, episode, 30000, cfg)
        original = StructuredMission(layout=cfg["layouts"][episode])
        expected = original.reset(cfg["training_map_seed_offset"] + 411000 + episode)
        assert stage == "full"
        np.testing.assert_array_equal(obs, expected)
        np.testing.assert_array_equal(env.goal, original.goal)
        actions = np.array([[0.2, -0.3, 0.1], [-0.4, 0.1, 0.2], [0., 0., 0.3]])
        actual_transition = env.step(actions)
        expected_transition = original.step(actions)
        np.testing.assert_array_equal(actual_transition[0], expected_transition[0])
        assert actual_transition[1:] == expected_transition[1:]


def test_short_stage_preserves_original_transition_rules():
    cfg = config()
    env, obs, stage = make_episode(422, 0, 0, cfg)
    original = StructuredMission(layout=cfg["layouts"][0])
    original.reset(cfg["training_map_seed_offset"] + 422000)
    for name in ("p", "boxes", "dynamic", "dv", "dr", "v"):
        np.testing.assert_array_equal(getattr(env, name), getattr(original, name))
    original.goal = env.goal.copy()
    np.testing.assert_array_equal(obs, original.observe())
    action = np.array([[0.5, 0.2, 0.], [0., -0.4, 0.3], [0.1, 0.2, -0.3]])
    actual_transition = env.step(action)
    expected_transition = original.step(action)
    np.testing.assert_array_equal(actual_transition[0], expected_transition[0])
    assert actual_transition[1:] == expected_transition[1:]
