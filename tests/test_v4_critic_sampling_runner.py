import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.train_v4_critic_sampling import (enrich_evaluation, new_stats,
    restore_diagnostic_snapshot, save_snapshot, update_stats)
from scripts.continue_masac_v3 import sha
from uav_planning.critic_sampling_diagnostic import CriticSamplingUpdater
from uav_planning.shared_masac import SharedMASAC
from uav_planning.structured import StructuredMission


@pytest.mark.parametrize('arm', ['uniform', 'stratified'])
def test_aligned_runner_snapshot_reproduces_next_action_and_updates(tmp_path, arm):
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/masac_v3/protocol.json').read_text())
    cfg.update(steps=128,warmup=0,replay_capacity=128,smoke=False,arms=['standard'],seeds=[411])
    torch.set_num_threads(1);torch.manual_seed(411)
    m=SharedMASAC(capacity=128,seed=411)
    env=StructuredMission(layout='urban');obs=env.reset(511000)
    exploration=np.random.default_rng(411)
    for i in range(128):
        action=np.zeros((3,3),np.float32)
        nxt,reward,done,trunc,_=env.step(action)
        assert not done and not trunc
        m.push(obs,action,reward,nxt,i%4==0);obs=nxt
    updater=CriticSamplingUpdater(m,arm=arm,seed=411,batch_size=128,done_quota=32)
    updater.update()
    stats=new_stats(env)
    save_snapshot(tmp_path,m,updater,env,exploration,step=128,episode=0,episode_return=0.,
                  episode_start=0,episode_stats=stats,partial_episode=False,parent_cfg=cfg,seed=411,arm=arm)

    def advance(model,up,environment,observation,st):
        result=[]
        for k in range(3):
            action=up.act(observation);before=environment.p.copy()
            nxt,reward,done,trunc,info=environment.step(action)
            assert not done and not trunc
            up.push(observation,action,reward,nxt,k==1)
            update_stats(st,environment,before)
            metrics={key:value for key,value in up.update().items() if not key.endswith('_s')}
            result.append((action,nxt,reward,done,trunc,info,metrics,model.index))
            observation=nxt
        weights={name:{k:v.detach().clone() for k,v in getattr(model,name).state_dict().items()}
                 for name in ('actor','q','target')}
        return result,weights,up.state_dict(),st

    direct=advance(m,updater,env,obs,copy.deepcopy(stats))
    restored,restored_up,restored_env,restored_obs,_,cp=restore_diagnostic_snapshot(
        tmp_path/'snapshots/128',device='cpu')
    replayed=advance(restored,restored_up,restored_env,restored_obs,copy.deepcopy(cp['episode_stats']))
    for a,b in zip(direct[0],replayed[0]):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1]);assert a[2:]==b[2:]
    for group in direct[1]:
        for key in direct[1][group]:assert torch.equal(direct[1][group][key],replayed[1][group][key])
    assert direct[2]['critic_rng']==replayed[2]['critic_rng']
    assert direct[2]['actor_rng']==replayed[2]['actor_rng']
    np.testing.assert_array_equal(direct[2]['strata']['position'],replayed[2]['strata']['position'])
    for key in ('closest_goal_m','path_m'):
        np.testing.assert_array_equal(direct[3][key],replayed[3][key])


def test_evaluation_reports_per_uav_metrics_and_static_contact_speed(tmp_path):
    p=np.array([[[10.,20.,10.],[50.,50.,10.],[70.,70.,10.]],
                [[12.,20.,10.],[50.,50.,10.],[70.,70.,10.]]])
    goals=np.array([[20.,20.,10.],[60.,50.,10.],[80.,70.,10.]])
    buildings=np.array([[[11.,19.,0.],[11.1,21.,20.]]])
    dynamic=np.full((2,4,3),200.)
    np.savez_compressed(tmp_path/'eval_100_800001.npz',positions=p,goals=goals,
                        dynamic=dynamic,buildings=buildings,dynamic_radii=np.full(4,2.5))
    rows=[dict(map_seed=800001,success=False,static_collision=True,boundary_collision=False,
               dynamic_collision=False,uav_collision=False,time=.2)]
    enrich_evaluation(tmp_path,100,rows)
    assert rows[0]['path_m_by_uav']==pytest.approx([2.,0.,0.])
    assert rows[0]['terminal_contact_speed_mps_by_type']['static']==pytest.approx([10.])
    assert rows[0]['final_speed_mps_by_uav']==pytest.approx([10.,0.,0.])
    assert not rows[0]['timeout']


@pytest.mark.parametrize('part', ['protocol.json', 'state/checkpoint_123.pt', 'state/replay_final.npz'])
def test_recovery_rejects_tampered_checkpoint_replay_or_protocol(tmp_path, part):
    state=tmp_path/'state';state.mkdir()
    (tmp_path/'protocol.json').write_text('{}')
    (state/'checkpoint_123.pt').write_bytes(b'checkpoint')
    (state/'replay_final.npz').write_bytes(b'replay')
    manifest=dict(complete=True,arm='uniform',seed=411,step=123,device='cpu',
        checkpoint_sha256=sha(state/'checkpoint_123.pt'),replay_sha256=sha(state/'replay_final.npz'),
        protocol_sha256=sha(tmp_path/'protocol.json'))
    (tmp_path/'complete.json').write_text(json.dumps(manifest))
    with (tmp_path/part).open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='Incomplete, changed or wrong-device'):
        restore_diagnostic_snapshot(tmp_path,device='cpu')
