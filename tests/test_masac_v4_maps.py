import json
import numpy as np
import pytest
import torch

from uav_planning.environment import swept_distance
from uav_planning.shared_masac import Actor
from uav_planning.structured import StructuredMission, box_distance, segment_box_hit
from uav_planning.v4_environment import V4Mission
from uav_planning.v4_maps import FAMILIES, LEVELS, default_config, generate_scene
from uav_planning.v4_reference import geometric_path, joint_reference


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('level', LEVELS)
def test_scenes_reproduce_and_object_masks_fit(family, level):
    cfg = default_config(family, level)
    scene = generate_scene(cfg, 910101)
    same = generate_scene(cfg, 910101)
    assert scene.scene_hash == same.scene_hash
    assert scene.to_dict() == same.to_dict()
    env = V4Mission(cfg)
    obs = env.reset(910101)
    assert obs.shape == (3, 162) and np.isfinite(obs).all()
    assert np.all(obs[:,143:] == 0)
    assert np.all((obs[:,20:55].reshape(3,5,7)[:,:,-1] == 0) |
                  (obs[:,20:55].reshape(3,5,7)[:,:,-1] == 1))
    if level == 'L0':
        assert len(scene.dynamic_positions) == 0
        assert np.all(obs[:,111:143] == 0)
        assert len(scene.boxes) <= 8
        assert np.all(obs[:,55:111].reshape(3,8,7)[:,:,-1].sum(axis=1) == len(scene.boxes))
        assert np.all(obs[:,20:55].reshape(3,5,7)[:,:,-1].sum(axis=1) == 2)
    assert np.allclose(np.linalg.norm(scene.goals-scene.starts,axis=1), cfg.goal_distance)


def test_l0_empty_objects_and_3_to_6_agents():
    for n in (3,4,5,6):
        env = V4Mission(default_config('urban_canyon','L0',n_agents=n,density=0))
        obs = env.reset(17)
        assert obs.shape == (n,162)
        assert np.all(obs[:,55:143] == 0)
        assert np.all(obs[:,20:55].reshape(n,5,7)[:,:,-1].sum(axis=1) == n-1)


def test_more_than_four_dynamic_obstacles_use_four_masked_nearest_slots():
    env = V4Mission(default_config('dynamic_crossing','L3',dynamic_count=6,sensor_range=400.))
    obs = env.reset(19)
    assert len(env.dynamic) == 6
    assert np.all(obs[:,111:143].reshape(3,4,8)[:,:,-1].sum(axis=1) == 4)
    for i in range(3):
        distances = np.linalg.norm(env.dynamic-env.p[i],axis=1)-env.dr
        ids = np.lexsort((np.arange(6),distances))[:4]
        np.testing.assert_allclose(obs[i,111:143].reshape(4,8)[:,:3],
                                   (env.dynamic[ids]-env.p[i])/max(env.world),atol=1e-7)


def test_u_opening_and_overpass_are_not_solid_envelopes():
    u = generate_scene(default_config('courtyard','L2'), 910101)
    center, outside = np.array([120.,120.,10.]), np.array([190.,120.,10.])
    radius = u.config.uav_radius+u.config.safety_margin
    assert not segment_box_hit(center,outside,u.boxes[:,0]-radius,u.boxes[:,1]+radius).any()
    bridge = generate_scene(default_config('overpass','L2'), 910101)
    for altitude in (20.,70.):
        a,b = np.array([48.,120.,altitude]),np.array([192.,120.,altitude])
        assert not segment_box_hit(a,b,bridge.boxes[:,0]-1,bridge.boxes[:,1]+1).any()
    a,b = np.array([48.,120.,50.]),np.array([192.,120.,50.])
    assert segment_box_hit(a,b,bridge.boxes[:,0]-1,bridge.boxes[:,1]+1).any()


def test_invalid_endpoints_use_identical_inflated_aabb_as_collision(monkeypatch):
    cfg = default_config('urban_canyon','L0',density=0.)
    plain = generate_scene(cfg, 0)
    point = plain.starts[0]
    lo, hi = point+np.array([.9,.9,-.5]), point+np.array([1.9,1.9,.5])
    assert box_distance(point, lo[None,:], hi[None,:])[0] > cfg.uav_radius
    assert segment_box_hit(point, point, (lo-cfg.uav_radius)[None,:],
                           (hi+cfg.uav_radius)[None,:]).any()
    monkeypatch.setattr('uav_planning.v4_maps._geometry',
                        lambda config,rng: ([np.stack([lo,hi]).tolist()], ['corner']))
    with pytest.raises(ValueError, match='inflated static geometry'):
        generate_scene(cfg, 0)


def test_out_of_range_objects_are_not_actor_inputs():
    cfg = default_config('dynamic_crossing','L3',sensor_range=10.)
    env = V4Mission(cfg); before = env.reset(910101)
    assert not before[0,55:111].reshape(8,7)[:,-1].any()
    assert not before[0,111:143].reshape(4,8)[:,-1].any()
    env.boxes += np.array([1., 0., 0.])
    env.dynamic += np.array([0., 1., 0.])
    after = env.observe()
    np.testing.assert_array_equal(before[0], after[0])


def test_nondefault_scene_hash_and_configuration_are_explicit():
    a = generate_scene(default_config('urban_canyon','L0',goal_distance=40.), 7)
    b = generate_scene(default_config('urban_canyon','L0',goal_distance=50.), 7)
    assert a.scene_hash != b.scene_hash
    c = generate_scene(default_config('urban_canyon','L0',goal_distance=40.), 8)
    assert not np.array_equal(a.starts, c.starts)
    with pytest.raises(ValueError, match='sensor range'):
        default_config('urban_canyon','L0',sensor_range=10.)
    with pytest.raises(ValueError, match='static-only'):
        default_config('urban_canyon','L1',dynamic_count=1)


def test_v4_matches_v3_dynamics_and_reward_for_same_scene():
    old = StructuredMission(layout='urban')
    obs_old = old.reset(800001)
    cfg = default_config('urban_canyon','L3',sensor_range=400.,dynamic_motion='linear')
    env = V4Mission(cfg); env.reset(1)
    for name in ('p','goal','boxes','lo','hi','dynamic','dv','dr','v','prev_acc','captured'):
        setattr(env, name, getattr(old,name).copy())
    obs = env.observe()
    actor = Actor().eval()
    with torch.no_grad():
        a1,_ = actor(torch.as_tensor(obs_old), deterministic=True)
        a2,_ = actor(torch.as_tensor(obs), deterministic=True)
    torch.testing.assert_close(a1, a2, atol=1e-6, rtol=1e-6)
    rng = np.random.default_rng(24)
    for _ in range(8):
        action = rng.uniform(-1,1,(3,3))
        _, reward_old, done_old, trunc_old, info_old = old.step(action)
        _, reward, done, trunc, info = env.step(action)
        np.testing.assert_allclose(env.p, old.p, atol=1e-12)
        np.testing.assert_allclose(env.v, old.v, atol=1e-12)
        assert reward == pytest.approx(reward_old, abs=1e-12)
        assert (done,trunc) == (done_old,trunc_old)
        for k in ('success','static_collision','dynamic_collision','uav_collision','boundary_collision'):
            assert info[k] == info_old[k]
        if done or trunc:
            break


def test_l0_reference_joint_rollout_replays_exactly():
    cfg = default_config('overpass','L0')
    result = joint_reference(cfg, 910101)
    assert result['joint_reference_success']
    assert len(result['attempts']) == len(result['traces'])
    trace = result['witness']
    assert len(trace['actions']) <= 350
    env = V4Mission(cfg); env.reset(910101)
    for k, action in enumerate(trace['actions']):
        assert np.linalg.norm(action, axis=1).max() <= 1+1e-12
        old_v = env.v.copy()
        _,_,done,trunc,info = env.step(action)
        np.testing.assert_allclose(env.p, trace['positions'][k+1], atol=1e-12)
        assert np.linalg.norm((env.v-old_v)/env.dt,axis=1).max() <= env.amax+1e-9
        assert np.linalg.norm(env.v,axis=1).max() <= env.vmax+1e-9
        if done or trunc:
            break
    assert info['success'] and env.t <= 350


def test_dynamic_crossing_logs_failed_and_successful_joint_attempts():
    cfg = default_config('dynamic_crossing','L3')
    result = joint_reference(cfg, 910101)
    assert result['attempts'][0]['final_info']['dynamic_collision']
    assert not result['attempts'][0]['success']
    assert result['joint_reference_success']
    assert result['attempts'][-1]['success']
    assert len(result['traces']) == len(result['attempts'])
    assert all(a['time_s'] <= 70 for a in result['attempts'])


def test_geometry_finds_path_and_does_not_claim_trajectory_before_execution():
    scene = generate_scene(default_config('courtyard','L2'), 910101)
    for agent in range(3):
        path, result = geometric_path(scene, agent)
        assert path is not None and result['status'].startswith('connected_')
        json.dumps(result, allow_nan=False)
        inflate = scene.config.uav_radius+scene.config.safety_margin
        for a,b in zip(path[:-1],path[1:]):
            assert not segment_box_hit(a,b,scene.boxes[:,0]-inflate,scene.boxes[:,1]+inflate).any()
