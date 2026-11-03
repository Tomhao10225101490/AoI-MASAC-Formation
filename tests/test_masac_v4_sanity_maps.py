"""Sanity map builders + short SharedMASAC smoke (no long train)."""
import json
from pathlib import Path

import numpy as np
import pytest

from uav_planning.shared_masac import SharedMASAC
from uav_planning.v4_sanity_maps import (
    SANITY_KINDS,
    make_sanity_env,
    straight_line_clear,
    write_fixture_catalog,
)


@pytest.mark.parametrize('kind', SANITY_KINDS)
def test_sanity_map_build_shapes_and_feasibility(kind):
    env, obs = make_sanity_env(kind, seed=870001 if kind == 'empty' else 870101)
    assert obs.shape == (3, 162) and np.isfinite(obs).all()
    assert env.config.level == 'L0'
    assert np.allclose(np.linalg.norm(env.goal - env.p, axis=1), env.config.goal_distance)
    if kind == 'empty':
        assert len(env.boxes) == 0
        assert straight_line_clear(env)
        assert np.all(obs[:, 55:111] == 0)
    else:
        assert len(env.boxes) == 1
        # Starts/goals must remain outside the inflated box (inject validates).
        assert env.scene.object_ids == ['sanity-single-obstacle']


def test_empty_reproducible_hash():
    a, _ = make_sanity_env('empty', 870002)
    b, _ = make_sanity_env('empty', 870002)
    assert a.scene.scene_hash == b.scene.scene_hash
    c, _ = make_sanity_env('empty', 870003)
    assert a.scene.scene_hash != c.scene.scene_hash


def test_single_obstacle_reproducible_and_distinct():
    a, _ = make_sanity_env('single_obstacle', 870102)
    b, _ = make_sanity_env('single_obstacle', 870102)
    assert a.scene.scene_hash == b.scene.scene_hash
    assert np.allclose(a.boxes, b.boxes)
    c, _ = make_sanity_env('single_obstacle', 870103)
    assert a.scene.scene_hash != c.scene.scene_hash


def test_fixture_catalog_roundtrip(tmp_path):
    path, catalog = write_fixture_catalog(tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded['version'] == 'masac-v4-sanity-maps-v1'
    assert len(loaded['maps']) == 8
    assert sum(1 for m in loaded['maps'] if m['kind'] == 'empty') == 4
    assert sum(1 for m in loaded['maps'] if m['kind'] == 'single_obstacle') == 4
    assert catalog == loaded


def test_short_shared_masac_smoke_on_empty():
    """One short unroll + a few updates; not a learning claim."""
    env, obs = make_sanity_env('empty', 870001)
    model = SharedMASAC(n=3, capacity=2048, seed=0)
    for t in range(64):
        action = model.act(obs)
        next_obs, reward, done, truncated, info = env.step(action)
        model.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        if model.size >= 32:
            losses = model.update(batch=16)
            assert all(np.isfinite(v) for v in losses.values())
        if done or truncated:
            env, obs = make_sanity_env('empty', 870001 + t + 1)
    assert model.size >= 32
