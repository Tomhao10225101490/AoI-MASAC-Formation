import copy
import json
import numpy as np
import pytest
import torch
from scripts.train_masac_v4_l0 import make_env, save_snapshot, restore_snapshot
from uav_planning.v4_training import InstrumentedLegacyMASAC


def advance(model, env, obs, explore, step):
    action=explore.uniform(-1,1,(3,3)).astype(np.float32) if step<=4 else model.act(obs)
    next_obs,reward,terminated,truncated,_=env.step(action)
    model.push(obs,action,reward,next_obs,terminated)
    loss=model.update(8) if step>4 else None
    return next_obs,action,loss


def test_complete_snapshot_exact_next_cpu_step_and_integrity(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(511)
    cfg={'sensor_range':400}; env=make_env('urban_canyon',cfg); obs=env.reset(7110000)
    model=InstrumentedLegacyMASAC(capacity=32,seed=511); explore=np.random.default_rng(511)
    for step in range(1,13): obs,_,_=advance(model,env,obs,explore,step)
    folder=save_snapshot(tmp_path,model,env,explore,{'step':12,'seed':511})
    expected_exploration=explore.random(4)
    expected_obs,expected_action,expected_loss=advance(model,env,obs,explore,13)
    expected_actor=copy.deepcopy(model.actor.state_dict()); expected_q=copy.deepcopy(model.q.state_dict())
    restored,env2,obs2,explore2,state=restore_snapshot(folder,32,'cpu')
    assert np.array_equal(explore2.random(4),expected_exploration)
    actual_obs,actual_action,actual_loss=advance(restored,env2,obs2,explore2,13)
    assert state=={'step':12,'seed':511}
    assert np.array_equal(actual_obs,expected_obs) and np.array_equal(actual_action,expected_action)
    assert actual_loss==expected_loss
    assert all(torch.equal(expected_actor[k],v) for k,v in restored.actor.state_dict().items())
    assert all(torch.equal(expected_q[k],v) for k,v in restored.q.state_dict().items())
    assert np.array_equal(model.obs[:model.size],restored.obs[:restored.size])
    with (folder/'replay.npz').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError,match='digest mismatch'): restore_snapshot(folder,32,'cpu')
