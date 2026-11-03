import numpy as np
from uav_planning.environment import Mission,swept_distance

def test_crossing_detected_between_frames():
    assert swept_distance(np.array([[-2.,0,0]]),np.array([[2.,0,0]]))[0]<1e-8

def test_limits_observations_and_reproducibility():
    for n in (3,4,6):
        e=Mission(n=n);o,s=e.reset(19);p=e.p.copy();assert all(len(x)==70 for x in o)
        e.reset(19);assert np.array_equal(p,e.p)
        for _ in range(10):
            v=e.v.copy();e.step(np.ones((n,3)))
            assert np.max(np.linalg.norm(e.v,axis=1))<=e.vmax+1e-8
            assert np.max(np.linalg.norm(e.v-v,axis=1))<=e.amax*e.dt+1e-8

def test_same_reward_across_feature_arms():
    results=[]
    for mode in ('standard','distance','prediction','maneuver'):
        e=Mission(mode=mode);e.reset(42);results.append(e.step(np.ones((3,3)))[2])
    assert all(np.array_equal(results[0],r) for r in results)


def test_goal_capture_does_not_teleport_velocity():
    e=Mission();e.reset(9);e.arrived[:]=True;e.v[:]=[1.9,0,0]
    old=e.v.copy();e.step(np.zeros((3,3)))
    assert np.max(np.linalg.norm(e.v-old,axis=1))<=e.amax*e.dt+1e-8
