import copy
import numpy as np
import torch
from uav_planning.shared_masac import SharedMASAC
from uav_planning.structured import StructuredMission
from uav_planning.v4_training import InstrumentedLegacyMASAC


def test_instrumentation_preserves_legacy_update_and_rng():
    torch.set_num_threads(1); torch.manual_seed(71)
    original=SharedMASAC(capacity=256,seed=71)
    env=StructuredMission(); obs=env.reset(771)
    rng=np.random.default_rng(88)
    for _ in range(140):
        action=rng.uniform(-1,1,(3,3)).astype(np.float32)
        no,r,done,trunc,_=env.step(action);original.push(obs,action,r,no,done);obs=no
        if done or trunc:obs=env.reset(int(rng.integers(10000)))
    cp=copy.deepcopy(original.checkpoint())
    measured=InstrumentedLegacyMASAC(capacity=256,seed=71)
    measured.load_state(cp)
    for key in ('obs','next','actions','rewards','done'):
        getattr(measured,key)[:original.size]=getattr(original,key)[:original.size]
    torch.set_rng_state(cp['torch_rng']); old=original.update(128)
    after=torch.get_rng_state().clone()
    torch.set_rng_state(cp['torch_rng']); new=measured.update(128)
    for key,value in old.items(): assert value==new[key]
    assert torch.equal(after,torch.get_rng_state())
    for name in ('actor','q','target'):
        for key,value in getattr(original,name).state_dict().items():
            assert torch.equal(value,getattr(measured,name).state_dict()[key])
    assert new['critic_grad_post']<=5.00001 and new['actor_grad_post']<=5.00001
