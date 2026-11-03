import torch
from uav_planning.v4_maneuver_masac import ManeuverMASAC
from uav_planning.v4_prediction import ProbeConfig


def _obs(n=3):
    x = torch.zeros(n, 162)
    x[..., 9:12] = 0.5
    x[..., 18] = 0.5
    x[..., 19] = 0.4
    x[..., 20:27] = torch.tensor([0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 1.0])
    x[..., 55:62] = torch.tensor([0.2, 0.0, 0.0, 0.05, 0.05, 0.05, 1.0])
    return x.numpy()


def test_maneuver_modes_update_and_counts():
    cfg = ProbeConfig(world=(240.0, 240.0, 90.0))
    counts = None
    for mode in ('full', 'static', 'zero'):
        m = ManeuverMASAC(n=3, capacity=64, seed=7, probe_mode=mode, probe_config=cfg)
        if counts is None:
            counts = m.parameter_counts()
        else:
            assert m.parameter_counts() == counts
        o = _obs()
        a = m.act(o)
        assert a.shape == (3, 3)
        m.push(o, a, 0.1, o, False)
        m.push(o, a, 0.0, o, False)
        # fill enough for batch
        for _ in range(20):
            m.push(o, a, 0.0, o, False)
        stats = m.update(batch=8)
        assert all(k in stats for k in ('actor_loss', 'critic_loss', 'probe_ttc_mean'))


def test_reject_attention_set_mode():
    try:
        ManeuverMASAC(set_mode='attention')
        assert False, 'expected ValueError'
    except ValueError:
        pass
