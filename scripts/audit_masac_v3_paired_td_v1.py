#!/usr/bin/env python3
"""Fixed-sample offline V3 SAC target, TD and critic-gradient audit.

This script computes diagnostic backward passes on copies of saved critics;
it never constructs an optimizer or changes a checkpoint/replay file.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_planning.shared_masac import Actor, TeamQ  # noqa: E402

PROTOCOL = ROOT / 'docs/design/MASAC_V3_TD_GRADIENT_AUDIT_20261103.md'
SAMPLE_BUDGET = 512
OFFSETS = {'curriculum': 10000, 'original': 20000}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def choose_indices(done, seed, budget=SAMPLE_BUDGET):
    values = np.asarray(done)
    if values.ndim != 1 or not np.isin(values, (0., 1.)).all():
        raise ValueError('Replay done must be a vector of zeros and ones')
    rng = np.random.default_rng(seed)
    groups, counts = {}, {}
    for value in (0, 1):
        ids = np.flatnonzero(values == value)
        counts[value] = len(ids)
        if not len(ids):
            raise ValueError(f'Empty done={value} population')
        groups[value] = rng.choice(ids, size=min(len(ids), budget), replace=False).astype(np.int64)
    return groups, counts


def summary(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return None
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite diagnostic values')
    return dict(n=int(len(x)), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), p10=float(np.percentile(x, 10)),
                median=float(np.median(x)), p90=float(np.percentile(x, 90)),
                p99=float(np.percentile(x, 99)), max=float(x.max()))


def sac_target(reward, done, minimum_target_q, alpha, team_mean_log_prob):
    return reward + .99 * (1. - done) * (minimum_target_q - alpha * team_mean_log_prob)


def group_gradient(critics, obs, actions, targets):
    """Two mean-MSE gradients with no clipping or optimizer; return flat vector."""
    if not len(obs):
        return None
    critics.zero_grad(set_to_none=True)
    estimates = [q(obs, actions) for q in critics]
    loss = sum(F.mse_loss(q, targets) for q in estimates)
    loss.backward()
    pieces = [p.grad.detach().reshape(-1).clone() for p in critics.parameters()]
    per_critic = [float(torch.linalg.vector_norm(torch.cat([p.grad.detach().reshape(-1) for p in q.parameters()])))
                  for q in critics]
    vec = torch.cat(pieces)
    critics.zero_grad(set_to_none=True)
    if not torch.isfinite(vec).all():
        raise ValueError('Nonfinite critic gradients')
    return dict(vector=vec, loss=float(loss.detach()), per_critic_norm=per_critic)


def mask_report(critics, o, a, target, q1, q2, near, capture, mask):
    ids = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long)
    if not len(ids):
        return dict(n=0, goal_capture_observations=0, target=None, absolute_td_q1=None,
                    absolute_td_q2=None, critic_gradient_norm=None)
    t, x1, x2 = target[ids], q1[ids], q2[ids]
    probe = group_gradient(critics, o[ids], a[ids], t)
    return dict(n=int(len(ids)), goal_capture_observations=int(np.count_nonzero(capture[mask])),
                target=summary(t.numpy()), absolute_td_q1=summary(torch.abs(t-x1).numpy()),
                absolute_td_q2=summary(torch.abs(t-x2).numpy()),
                critic_gradient_norm=float(torch.linalg.vector_norm(probe['vector'])))


def verify_sources(root):
    marker = json.loads((root / 'completion.json').read_text())
    if not marker.get('complete') or marker.get('runs') != 6 or marker.get('smoke') or (root / 'error.json').exists():
        raise ValueError('Expected completed error-free six-run paired batch')
    manifest = json.loads((root / 'source_hashes.json').read_text())
    result = {}
    for name in ('shared_masac.py', 'structured.py'):
        expected = manifest[name]
        snapshot, live = root / 'source_snapshot' / name, ROOT / 'uav_planning' / name
        if sha(snapshot) != expected or sha(live) != expected:
            raise ValueError(f'Live or snapshot source mismatch: {name}')
        result[name] = expected
    return result


def run_one(root, arm, seed):
    snapshot = root / f'{arm}_{seed}' / 'snapshots/300000'
    manifest = json.loads((snapshot / 'complete.json').read_text())
    checkpoint = snapshot / 'state/checkpoint_300000.pt'
    replay_file = snapshot / 'state/replay_final.npz'
    config = snapshot / 'protocol.json'
    if (not manifest.get('complete') or manifest['arm'] != arm or manifest['seed'] != seed or
            manifest['step'] != 300000 or sha(checkpoint) != manifest['checkpoint_sha256'] or
            sha(replay_file) != manifest['replay_sha256'] or sha(config) != manifest['protocol_sha256']):
        raise ValueError(f'Invalid complete snapshot: {snapshot}')
    cfg = json.loads(config.read_text())
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (cp['step'] != 300000 or cp['seed'] != seed or cp['replay_size'] != 100000 or
            cp['replay_index'] != 0 or cp['cfg'] != cfg or cp.get('history_arm') != arm or
            not cp.get('exact_resume') or cfg['replay_capacity'] != 100000 or cfg['gamma'] != .99 or
            cfg['n_agents'] != 3 or cfg['vmax'] != 10.):
        raise ValueError(f'Invalid checkpoint/config alignment: {snapshot}')
    torch.manual_seed(0)  # Initialization draws are replaced by checkpoint weights.
    actor = Actor()
    critics = nn.ModuleList([TeamQ(), TeamQ()])
    targets = nn.ModuleList([TeamQ(), TeamQ()])
    actor.load_state_dict(cp['actor'])
    critics.load_state_dict(cp['critics'])
    targets.load_state_dict(cp['targets'])
    actor.requires_grad_(False)
    targets.requires_grad_(False)
    alpha = cp['logalpha'].detach().exp()
    numpy_seed = 20261103 + seed + OFFSETS[arm]
    with np.load(replay_file, allow_pickle=False) as replay:
        done = replay['done']
        if done.shape != (100000,):
            raise ValueError('Wrong done shape')
        indices, populations = choose_indices(done, numpy_seed)
        selected = np.concatenate([indices[0], indices[1]])
        arrays = {}
        expected = {'obs': (100000, 3, 162), 'next_obs': (100000, 3, 162),
                    'actions': (100000, 3, 3), 'rewards': (100000,)}
        for key, shape in expected.items():
            values = replay[key]
            if values.shape != shape or values.dtype != np.float32:
                raise ValueError(f'Unexpected replay shape or dtype for {key}')
            arrays[key] = np.array(values[selected], copy=True)
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f'Nonfinite sampled replay values: {key}')
            del values
    results, gradients = {}, {}
    start = 0
    for value in (0, 1):
        ids = indices[value]
        stop = start + len(ids)
        o, no, a, reward = [torch.from_numpy(arrays[k][start:stop])
                            for k in ('obs', 'next_obs', 'actions', 'rewards')]
        d = torch.full_like(reward, float(value))
        action_seed = 987654321 + seed + OFFSETS[arm] + value
        torch.manual_seed(action_seed)
        with torch.no_grad():
            next_action, log_prob = actor(no)
            qtarget = [q(no, next_action) for q in targets]
            minimum = torch.minimum(*qtarget)
            mean_log_prob = log_prob.mean(-1)
            target = sac_target(reward, d, minimum, alpha, mean_log_prob)
            q1, q2 = [q(o, a) for q in critics]
            bootstrap = .99 * (1. - d) * (minimum - alpha * mean_log_prob)
            entropy_term = -alpha * mean_log_prob
        if value == 1 and not torch.equal(target, reward):
            raise AssertionError('Terminal target must equal reward')
        probe = group_gradient(critics, o, a, target)
        gradients[value] = probe.pop('vector')
        qnorm = float(torch.linalg.vector_norm(gradients[value]))
        distances = np.linalg.norm(o.numpy()[..., :3] * 240., axis=-1)
        speeds = np.linalg.norm(o.numpy()[..., 3:6] * cfg['vmax'], axis=-1)
        near = (distances < 50.).any(axis=1)
        capture = ((distances < 4.) & (speeds < 2.)).any(axis=1)
        results[str(value)] = dict(done=value, population=populations[value],
            population_fraction=populations[value]/100000., sampled=len(ids),
            sampling_without_replacement=True,
            sampled_indices_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
            index_encoding='signed little-endian int64 in sampling order',
            torch_action_seed=action_seed, reward=summary(reward.numpy()),
            target=summary(target.numpy()), bootstrap=summary(bootstrap.numpy()),
            alpha_times_negative_mean_log_prob=summary(entropy_term.numpy()),
            next_min_target_q=summary(minimum.numpy()), online_q1=summary(q1.numpy()),
            online_q2=summary(q2.numpy()), absolute_q_disagreement=summary(torch.abs(q1-q2).numpy()),
            signed_td_q1=summary((target-q1).numpy()), signed_td_q2=summary((target-q2).numpy()),
            absolute_td_q1=summary(torch.abs(target-q1).numpy()),
            absolute_td_q2=summary(torch.abs(target-q2).numpy()),
            critic_loss_sum_mse=probe['loss'], combined_critic_preclip_gradient_norm=qnorm,
            critic_preclip_gradient_norm_by_q=probe['per_critic_norm'],
            hypothetical_training_clip_scale=min(1.,5./qnorm) if qnorm else 1.,
            near_any_agent_lt50m=mask_report(critics, o, a, target, q1, q2, near, capture, near),
            far_all_agents_ge50m=mask_report(critics, o, a, target, q1, q2, near, capture, ~near))
        start = stop
    w0, w1 = populations[0]/100000., populations[1]/100000.
    v0, v1 = gradients[0], gradients[1]
    weighted0, weighted1 = w0*v0, w1*v1
    denom = float(torch.linalg.vector_norm(v0)*torch.linalg.vector_norm(v1))
    combined = weighted0+weighted1
    n = float(torch.linalg.vector_norm(combined))
    wmean = lambda key: w0*results['0'][key]['mean'] + w1*results['1'][key]['mean']
    return dict(arm=arm, seed=seed, checkpoint_sha256=manifest['checkpoint_sha256'],
        replay_sha256=manifest['replay_sha256'], snapshot_protocol_sha256=manifest['protocol_sha256'],
        numpy_sampling_seed=numpy_seed, alpha=float(alpha), strata=results,
        population_weighted_estimate=dict(
            expected_reward=wmean('reward'), expected_target=wmean('target'),
            expected_absolute_td_q1=wmean('absolute_td_q1'), expected_absolute_td_q2=wmean('absolute_td_q2'),
            gradient_norm_nondone_contribution=float(torch.linalg.vector_norm(weighted0)),
            gradient_norm_done_contribution=float(torch.linalg.vector_norm(weighted1)),
            combined_gradient_norm=n,
            hypothetical_training_clip_scale=min(1.,5./n) if n else 1.,
            strata_gradient_cosine=float(torch.dot(v0,v1)/denom) if denom else None,
            note='Weighted gradient vector contributions may reinforce or cancel; norms are not additive.'))


def run(root):
    hashes = verify_sources(root)
    torch.set_num_threads(1)
    rows = [run_one(root, arm, seed) for arm in ('curriculum','original') for seed in (411,422,433)]
    return dict(protocol_sha256=sha(PROTOCOL), audit_script_sha256=sha(Path(__file__)),
        source_sha256=hashes, run=str(root.resolve()), torch_version=torch.__version__,
        numpy_version=np.__version__, cpu_threads=torch.get_num_threads(),
        gamma=.99, per_stratum_max_sample=SAMPLE_BUDGET,
        semantics='done means collision or all-agent success; horizon truncation alone is not stored as done and is bootstrapped. Replay has no collision type or distinct truncation field.',
        limitations='Offline correlated replay sample, one stochastic next action per transition, terminal/nonterminal samples weighted by replay population; gradients are diagnostic un-clipped backward passes, not training updates or a causal attribution.',
        rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = run(args.run.resolve())
    with args.out.open('x') as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write('\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
