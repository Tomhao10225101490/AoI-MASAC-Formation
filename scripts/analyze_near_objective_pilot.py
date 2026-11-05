#!/usr/bin/env python3
"""Independent completed-pilot integrity/count analysis; no model loading or new rollout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def analyze(mode):
    root = ROOT / f'results/masac_v4_near_objective_2k_{mode}_cuda_20261105'
    directory = root / 'maneuver_static_1644'
    manifest, completion = read(root / 'run_manifest.json'), read(root / 'completion.json')
    final, cp = read(directory / 'completion.json'), directory / 'checkpoint_002000'
    snap = read(cp / 'manifest.json')
    config = ROOT / f'configs/masac_v4/near_objective_2k_{mode}_cuda_20261105.json'
    require(completion['complete'] and final['step'] == 2000 and final['seed'] == 1644,
            'Incomplete or wrong population')
    require(manifest['config_hashes']['protocol'] == sha(config) and
            manifest['near_objective']['mode'] == mode and len(manifest['sources']) == 17,
            'Config/entry-point provenance mismatch')
    require(snap['complete'] and snap['step'] == 2000 and snap['replay_size'] == 2000 and
            snap['sources'] == manifest['sources'], 'Invalid final snapshot')
    for path, expected in manifest['sources'].items():
        require(sha(ROOT / path) == expected, f'Frozen source changed: {path}')
    require(sha(cp / 'checkpoint.pt') == snap['checkpoint_sha256'] and
            sha(cp / 'replay.npz') == snap['replay_sha256'], 'Final artifact hash mismatch')
    records = rows(root / 'near_reward_transitions.jsonl')
    require(len(records) == 2000 and [r['training_step'] for r in records] == list(range(1, 2001)),
            'Missing/duplicate reward transition')
    for i, r in enumerate(records):
        before, after = r['previous_phi'], r['following_phi_observed']
        effective = 0 if r['terminated'] else after
        expected = 0. if mode == 'original' else .99 * effective - before - after
        require(0 <= before <= .4 and 0 <= after <= .4 and
                np.isclose(r['correction'], expected, rtol=0, atol=1e-12) and
                np.isclose(r['replay_reward'], r['legacy_base_reward'] + expected, rtol=0, atol=1e-12),
                'Reward conversion mismatch')
        if mode == 'original':
            require(r['replay_reward'] == r['legacy_base_reward'] and r['removed_near'] == 0 and
                    r['potential_difference'] == 0, 'Original numerical reward changed')
        else:
            require(r['removed_near'] == after and
                    np.isclose(r['potential_difference'], .99 * effective - before, atol=1e-12, rtol=0),
                    'Replacement component mismatch')
        if i and not (records[i-1]['terminated'] or records[i-1]['truncated']):
            require(before == records[i-1]['following_phi_observed'] and
                    r['map_seed'] == records[i-1]['map_seed'], 'Potential state continuity mismatch')
    with np.load(cp / 'replay.npz', allow_pickle=False) as data:
        require(np.allclose(data['rewards'].reshape(-1), [r['replay_reward'] for r in records],
                            atol=2e-6, rtol=1e-7), 'Replay does not contain transformed reward')
        require(np.array_equal(data['done'].reshape(-1).astype(bool), [r['terminated'] for r in records]),
                'Replay terminal mask differs')
    metrics = rows(directory / 'metrics.jsonl')
    progress = [r for r in metrics if r.get('kind') == 'progress' and 'actor_loss' in r]
    finite_fields = ('actor_loss', 'critic_loss', 'alpha', 'actor_grad', 'critic_grad',
                     'actor_grad_post', 'critic_grad_post', 'q_mean', 'target_mean',
                     'td_abs', 'entropy', 'terminal_sample_fraction', 'probe_clearance_mean',
                     'probe_ttc_mean', 'probe_positive_preview_frac')
    require(progress and progress[-1]['step'] == 2000 and progress[-1]['updates'] == 1000 and
            all(np.isfinite([r[k] for k in finite_fields]).all() for r in progress),
            'Missing final update/finite diagnostics')
    episodes = [r for r in metrics if r.get('kind') == 'episode']
    start = 0
    for episode in episodes:
        end = episode['step']
        segment = records[start:end]
        for key, component in (('near_removed_return', 'removed_near'),
                               ('near_potential_difference_return', 'potential_difference'),
                               ('shaping_return', 'correction'), ('base_return', 'legacy_base_reward')):
            require(np.isclose(episode[key], sum(r[component] for r in segment), atol=1e-9, rtol=1e-9),
                    f'Episode component accumulation mismatch: {key}')
        start = end
    evaluation = read(directory / 'evaluation_2000.json')
    require(len(evaluation) == 12 and {r['map_seed'] for r in evaluation} == set(range(830001, 830013)) and
            all(r['evaluation_shaping_enabled'] is False for r in evaluation), 'Invalid development evaluation')
    collisions = ('uav_collision', 'static_collision', 'dynamic_collision', 'boundary_collision')
    failures = Counter()
    for r in evaluation:
        if r['success']:
            failures['success'] += 1
        elif any(r[k] for k in collisions):
            for k in collisions:
                if r[k]:
                    failures[k] += 1
        else:
            failures['timeout_after_ball_all' if r['ever_ball_all'] else
                     'timeout_after_ball_any_only' if r['ever_ball_any'] else 'timeout_before_ball'] += 1
    require('NEAR-EVAL-STATE PASS' in (root / 'train.log').read_text(), 'Missing evaluation-state guard')
    result = dict(mode=mode, complete=True, step=2000, updates=1000, reward_transitions=2000,
                  finite_sampled_progress_rows=len(progress), source_hashes_verified=17,
                  checkpoint_sha256=snap['checkpoint_sha256'], replay_sha256=snap['replay_sha256'],
                  successes=sum(r['success'] for r in evaluation),
                  ever_ball_any=sum(r['ever_ball_any'] for r in evaluation),
                  ever_ball_all=sum(r['ever_ball_all'] for r in evaluation),
                  failure_counts_nonexclusive=dict(failures),
                  nonzero_potential_transitions=sum(r['previous_phi'] != 0 or
                                                  r['following_phi_observed'] != 0 for r in records),
                  nonzero_corrections=sum(r['correction'] != 0 for r in records),
                  terminal_transitions=sum(r['terminated'] for r in records),
                  timeout_transitions=sum(r['truncated'] and not r['terminated'] for r in records),
                  min_correction=min(r['correction'] for r in records),
                  max_correction=max(r['correction'] for r in records),
                  elapsed_s=completion['elapsed_s'], evaluation_training_state_guard='PASS')
    print('PILOT ' + json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    out = p.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    try:
        arms = [analyze(mode) for mode in ('original', 'potential')]
    except Exception:
        (out / 'error.json').write_text(json.dumps(dict(complete=False, traceback=traceback.format_exc()),
                                                 indent=2)+'\n', encoding='utf-8')
        raise
    result = dict(runtime_integrity='PASS', arms=arms,
                  scope='Single-seed2k numerical/runtime sanity; no gain or advancement decision.',
                  intervention_exercised=arms[1]['nonzero_corrections'] > 0)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (out / 'completion.json').write_text(json.dumps(dict(complete=True, analysis_sha256=sha(out/'analysis.json'),
                                                        analyzer_sha256=sha(Path(__file__))), indent=2)+'\n', encoding='utf-8')
    print('COMPLETE runtime_integrity=PASS intervention_exercised=' + str(result['intervention_exercised']), flush=True)


if __name__ == '__main__':
    main()
