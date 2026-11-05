#!/usr/bin/env python3
"""Verify and summarize completed dwell artifacts; no new trajectory/policy evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    if not path.is_file():
        return 'UNAVAILABLE'
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(value, message):
    if not value:
        raise ValueError(message)


def population(agents):
    return dict(agent_maps=len(agents),
                agent_frames=sum(sum(b['frames'] for b in a['bands'].values()) for a in agents),
                near_stall_frames=sum(a['near_stall']['frames'] for a in agents),
                with_25plus_run=sum(bool(a['near_stall_runs_25plus']) for a in agents),
                longest_run_frames=max((a['near_stall']['longest_frames'] for a in agents), default=0),
                final_near_stall=sum(4 <= a['final_distance_m'] < 10 and a['final_speed_mps'] < .5
                                     for a in agents))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root, out = args.audit.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    completion, manifest, summary = [load(root / n) for n in ('completion.json', 'manifest.json', 'summary.json')]
    require(completion['complete'] and completion['dwell_available_maps'] == 72, 'Audit incomplete')
    for name, expected in completion['artifact_sha256'].items():
        require(sha(root / name) == expected, f'Output hash mismatch: {name}')
    for path, expected in manifest['inputs_sha256'].items():
        require(sha(Path(path)) == expected, f'Input hash mismatch: {path}')
    require(manifest['input_hashes_rechecked'] and manifest['original_training_source_config_hashes_verified'],
            'Freeze not verified')
    records = [json.loads(line) for line in (root / 'maps.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(records) == 72 and records == [m for h in summary['histories'] for m in h['maps']],
            'Map records/summary disagree')
    require(sum(line.startswith('MAP ') for line in (root / 'audit.log').read_text().splitlines()) == 72,
            'Missing map progress lines')
    require(all(m['saved_physics_cross_check']['status'] == 'PASS' and
                m['reward']['reference_validation'] == 'UNAVAILABLE' for m in records), 'Invalid scope/status')
    max_v_error = max(m['saved_physics_cross_check']['max_velocity_abs_error'] for m in records)
    max_p_error = max(m['saved_physics_cross_check']['max_position_abs_error'] for m in records)
    histories = []
    for history in summary['histories']:
        maps = history['maps']
        agents = [a for m in maps for a in m['agents']]
        require(len(maps) == 12 and len(agents) == 36, 'Unexpected history population')
        for m in maps:
            components = m['reward']['component_totals']
            for key in ('undiscounted_team_mean', 'raw_discounted_team_mean'):
                require(np.isclose(components['total'][key], sum(c[key] for n, c in components.items()
                                                               if n != 'total'), atol=1e-10), 'Component sum mismatch')
        classes = {name: population([a for a in agents if a['classification'] == name])
                   for name in ('never_ball', 'ball_without_capture', 'ever_capture')}
        total = population(agents)
        require(total['agent_frames'] == history['agent_frames'] and
                total['near_stall_frames'] == history['near_stall_agent_frames'] and
                total['with_25plus_run'] == history['agent_maps_with_25plus_run'], 'History aggregation mismatch')
        rewards = {}
        for name in maps[0]['reward']['component_totals']:
            rewards[name] = {key: float(np.mean([m['reward']['component_totals'][name][key] for m in maps]))
                             for key in ('undiscounted_team_mean', 'raw_discounted_team_mean')}
        h = dict(arm=history['arm'], seed=history['training_seed'], **total, classes=classes,
                 successes=sum(m['json_success'] for m in maps),
                 failed_maps_with_any_25plus_run=sum(not m['json_success'] and any(
                     a['near_stall_runs_25plus'] for a in m['agents']) for m in maps),
                 band_frames=history['band_agent_frames'], mean_episode_reward_components=rewards,
                 capture_entries=sum(len(a['capture_entry_times_s']) for a in agents),
                 capture_exits=sum(len(a['capture_exit_times_s']) for a in agents))
        histories.append(h)
        print(f'HISTORY {h["arm"]} seed={h["seed"]} success={h["successes"]}/12 '
              f'near_stall={h["near_stall_frames"]}/{h["agent_frames"]} '
              f'25plus={h["with_25plus_run"]}/36 never_ball={classes["never_ball"]} '
              f'capture_entries/exits={h["capture_entries"]}/{h["capture_exits"]}', flush=True)
    never = population([a for m in records for a in m['agents'] if a['classification'] == 'never_ball'])
    require(never['agent_maps'] == 65, 'Previously reported never-ball total differs')
    result = dict(integrity='PASS', artifact_hashes_verified=len(completion['artifact_sha256']),
                  input_hashes_verified=len(manifest['inputs_sha256']), maps=72, agents=216,
                  max_velocity_abs_error=max_v_error, max_position_abs_error=max_p_error,
                  histories=histories, never_ball_population=never,
                  reward_reference_validation='UNAVAILABLE',
                  scope='Descriptive, correlated, policy-dependent lengths/terminal censoring; not causal arm comparison.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Final saved-trajectory dwell and reward results', '', result['scope'], '',
             'All 72 trajectories and frozen inputs verified. Formula-derived reward has no logged-return reference.', '',
             '| Arm | Seed | Agent-frames | Near-stall frames (%) | >=25-frame agent-maps /36 | Never-ball with run /total | Capture entries/exits |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for h in histories:
        n = h['classes']['never_ball']
        lines.append(f'| {h["arm"]} | {h["seed"]} | {h["agent_frames"]} | {h["near_stall_frames"]} '
                     f'({100*h["near_stall_frames"]/h["agent_frames"]:.2f}) | {h["with_25plus_run"]} | '
                     f'{n["with_25plus_run"]}/{n["agent_maps"]} | {h["capture_entries"]}/{h["capture_exits"]} |')
    lines += ['', 'Near-stall means 4<=d<10m and speed<0.5m/s on the sampled grid. '
              '25 samples represent 5s of occupancy count and a 4.8s endpoint span.', '',
              f'Never-ball population: {never}. This conditional population is not an independent arm comparison.', '',
              '| Arm | Seed | Progress | Near | Acceleration | Capture delta | Terminal | Total |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    names = ('progress', 'near', 'real_acceleration_penalty', 'capture_delta', 'team_terminal', 'total')
    for h in histories:
        r = h['mean_episode_reward_components']
        lines.append(f'| {h["arm"]} | {h["seed"]} | ' + ' | '.join(
            f'{r[n]["raw_discounted_team_mean"]:.5f}' for n in names) + ' |')
    lines += ['', 'Reward table: mean raw gamma=.99 discounted base component totals over all 12 saved episodes/history. '
              'Living/first-ball/collision and all undiscounted components remain in JSON. '
              'No entropy or bootstrap is included; reference validation is UNAVAILABLE.', '']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    bands = list(histories[0]['band_frames'])
    x = np.arange(6)
    bottom = np.zeros(6)
    for name, color in zip(bands, plt.get_cmap('tab20').colors):
        y = np.array([100*h['band_frames'][name]/h['agent_frames'] for h in histories])
        ax.bar(x, y, bottom=bottom, label=name.replace('_', ' '), color=color)
        bottom += y
    ax.set_xticks(x, [f'{h["arm"]}\n{h["seed"]}' for h in histories])
    ax.set_ylabel('Percent of recorded agent-frames')
    ax.set_ylim(0, 100)
    ax.set_title('Final40k development trajectories: sampled occupancy\nCorrelated frames; lengths and terminal censoring differ')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(out / 'occupancy.png', dpi=180)
    fig.savefig(out / 'occupancy.svg')
    plt.close(fig)
    svg = out / 'occupancy.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    (out / 'completion.json').write_text(json.dumps(dict(complete=True, artifact_sha256={
        n: sha(out / n) for n in ('summary.json', 'summary.md', 'occupancy.png', 'occupancy.svg')},
        input_completion_sha256=sha(root / 'completion.json'), script_sha256=sha(Path(__file__))), indent=2) + '\n', encoding='utf-8')
    print(f'INTEGRITY PASS inputs={len(manifest["inputs_sha256"])} outputs=6 physics_max_v={max_v_error} '
          f'physics_max_p={max_p_error} never_ball={never} out={out}', flush=True)


if __name__ == '__main__':
    main()
