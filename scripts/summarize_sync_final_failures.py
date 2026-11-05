#!/usr/bin/env python3
"""Reproduce post-hoc final40k failure tables and fixed-endpoint figure.

Reads only previously checked trajectory-summary JSONs; no policies or rollouts.
This script does not replace the registered study decision analyzer.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SEEDS = (1511, 1522, 1533)
ARMS = ('control', 'treatment')
CLASSES = ('success', 'incomplete_ball_arrival',
           'ball_without_all_individual_capture', 'individual_capture_without_joint', 'collision')
SCOPE = ('Post-hoc descriptive final40k development trajectory diagnosis; three training '
         'replicates, correlated agent/map observations; not a new efficacy endpoint or causal attribution')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('ARTIFACT', path.resolve(), flush=True)


def describe_never_ball(rows):
    if not rows:
        return {'n': 0}
    closest = np.array([r['post_step_min_distance_m'] for r in rows])
    final_distance = np.array([r['final_distance_m'] for r in rows])
    final_speed = np.array([r['final_speed_mps'] for r in rows])
    assert all(np.isfinite(v).all() for v in (closest, final_distance, final_speed))
    assert (closest >= 4).all()
    return {'n': len(rows),
            'closest_quantiles_0_25_50_75_100_m': np.quantile(closest, [0, .25, .5, .75, 1]).tolist(),
            'closest_4_to_5_m': int(((closest >= 4) & (closest < 5)).sum()),
            'closest_5_to_10_m': int(((closest >= 5) & (closest < 10)).sum()),
            'closest_at_least_10_m': int((closest >= 10).sum()),
            'final_speed_below_2_mps': int((final_speed < 2).sum()),
            'final_distance_quantiles_0_25_50_75_100_m': np.quantile(final_distance, [0, .25, .5, .75, 1]).tolist()}


def aggregate(progress, out):
    totals, pairs, distance_records = {}, [], []
    never_ball = {arm: [] for arm in ARMS}
    for seed in SEEDS:
        source = progress / f'trajectory_40000_seed{seed}/summary.json'
        data = json.loads(source.read_text(encoding='utf-8'))
        assert data['step'] == 40000 and data['training_seed'] == seed
        assert data['paired_map_metadata_verified'] and set(data['arms']) == set(ARMS)
        metadata = [[(r['map_seed'], r['family'], r['scene_sha256'])
                     for r in data['arms'][arm]['maps']] for arm in ARMS]
        assert metadata[0] == metadata[1] and len(set(metadata[0])) == 12
        pair = {'seed': seed, 'source': str(source), 'arms': {}}
        for arm, audit in data['arms'].items():
            counts = {k: 0 for k in (*CLASSES, 'any_ball', 'all_ball', 'all_individual_capture')}
            assert len(audit['maps']) == 12
            rows = []
            for row in audit['maps']:
                a, c, j = (row[k] for k in ('all_agents_ever_in_ball',
                                           'all_agents_ever_captured', 'recomputed_joint_capture'))
                assert (not c or a) and (not j or c)
                assert row['json_success'] == row['recomputed_collision_free_success']
                assert row['json_success'] == (j and not row['collision'])
                counts['any_ball'] += row['any_agent_ever_in_ball']
                counts['all_ball'] += a
                counts['all_individual_capture'] += c
                key = ('collision' if row['collision'] else 'success' if row['json_success']
                       else 'incomplete_ball_arrival' if not a
                       else 'ball_without_all_individual_capture' if not c
                       else 'individual_capture_without_joint')
                counts[key] += 1
                rows.extend(agent for agent in row['agents'] if agent['classification'] == 'never_ball')
            assert sum(counts[k] for k in CLASSES) == 12
            pair['arms'][arm] = counts
            totals.setdefault(arm, {k: 0 for k in counts})
            for key, value in counts.items():
                totals[arm][key] += value
            never_ball[arm].extend(rows)
            distance_records.append({'seed': seed, 'arm': arm, **describe_never_ball(rows)})
            print('FINAL_FAILURE_COUNTS', seed, arm, json.dumps(counts), flush=True)
        pairs.append(pair)
    taxonomy = {'scope': SCOPE,
                'definitions': {'A': 'all agents ever distance<4m',
                                'C': 'all agents ever distance<4m and speed<2m/s',
                                'J': 'simultaneous strict capture; unlatched',
                                'hierarchy': 'J implies C implies A'},
                'pairs': pairs, 'totals': totals}
    distances = {'scope': SCOPE, 'selection': 'classification never_ball using strict post-step distance<4m',
                 'closest_bins_m': ['[4,5)', '[5,10)', '[10,infinity)'],
                 'records': distance_records,
                 'totals': {arm: describe_never_ball(rows) for arm, rows in never_ball.items()}}
    save(out / 'failure_taxonomy.json', taxonomy)
    save(out / 'distance_shortfall_diagnosis.json', distances)
    return taxonomy


def plot(data, out):
    assert all(data['totals'][a]['collision'] == 0 for a in ARMS), 'Figure requires adding a collision class'
    control = np.array([p['arms']['control']['success'] for p in data['pairs']])
    b = np.array([p['arms']['treatment']['success'] for p in data['pairs']])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    x = np.arange(3)
    for i, (c, t) in enumerate(zip(control, b)):
        axes[0].plot([i-.12, i+.12], [c, t], color='#777777', linewidth=1.5)
        axes[0].annotate(f'{t-c:+d}', (i, max(c, t)+.6), ha='center', fontsize=10)
    axes[0].scatter(x-.12, control, s=55, color='#0072B2', label='Static control', zorder=3)
    axes[0].scatter(x+.12, b, s=55, color='#D55E00', marker='s', label='Static+B', zorder=3)
    axes[0].axhline(8, color='#555555', linestyle='--', linewidth=1, label='Per-seed gate: 8/12')
    axes[0].set(xticks=x, xticklabels=SEEDS, yticks=np.arange(0, 13, 2), ylim=(-.5, 12.3),
                xlabel='Paired training seed', ylabel='Final joint successes / 12 maps',
                title='(a) All registered 40k endpoints')
    axes[0].grid(axis='y', alpha=.2)
    axes[0].legend(loc='upper left', fontsize=8)
    labels = ('Success', 'Not all enter ball', 'All ball, not all individual capture',
              'Individual capture, no overlap')
    colors = ('#009E73', '#CC79A7', '#E69F00', '#56B4E9')
    bottom = np.zeros(2)
    for key, label, color in zip(CLASSES[:-1], labels, colors):
        counts = np.array([data['totals'][a][key] for a in ARMS])
        axes[1].bar([0, 1], counts, bottom=bottom, label=label, color=color, width=.6)
        for i, n in enumerate(counts):
            if n:
                axes[1].text(i, bottom[i]+n/2, str(n), ha='center', va='center', fontsize=10)
        bottom += counts
    axes[1].set(xticks=[0, 1], xticklabels=['Static control', 'Static+B'], ylim=(0, 38),
                yticks=np.arange(0, 37, 6), ylabel='Descriptive episodes / 36 per arm',
                title='(b) Exhaustive final-event classes')
    axes[1].legend(loc='lower center', bbox_to_anchor=(.5, -.41), fontsize=8, ncol=2)
    fig.suptitle('Three training pairs; reused development maps; both absolute gates FAIL', fontsize=11)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    for suffix in ('png', 'svg'):
        path = out / f'final_endpoints_and_failure_classes.{suffix}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        if suffix == 'svg':
            text = path.read_text(encoding='utf-8')
            path.write_text('\n'.join(line.rstrip() for line in text.splitlines()) + '\n', encoding='utf-8')
        print('FIGURE', path.resolve(), flush=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot(aggregate(args.progress, args.out), args.out)


if __name__ == '__main__':
    main()
