#!/usr/bin/env python3
"""Read-only partial/final analysis of the preregistered 40k static B study.

Reads manifests, completion markers, metrics and development evaluations only.
Never loads model weights, launches a policy, or opens the sealed test manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

SEEDS = (1511, 1522, 1533)
FINAL_STEP = 40000
MAPS = 12
EVAL_NAME = re.compile(r'^evaluation_(\d+)\.json$')
COLLISIONS = ('uav_collision', 'static_collision', 'dynamic_collision', 'boundary_collision')
LOSS_KEYS = ('actor_loss', 'critic_loss', 'alpha', 'entropy')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def optional_json(path, issues):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError) as exc:
        issues.append(f'{path}: cannot read JSON: {exc}')
        return None


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values):
    return sum(values) / len(values) if values else None


def failure_reason(row):
    if row.get('success') is True:
        return 'success'
    for key in COLLISIONS:
        if row.get(key):
            return key
    if row.get('timeout'):
        ball = row.get('ever_in_ball_by_uav') or []
        if ball and all(ball):
            return 'timeout_after_ball_all'
        return 'timeout_after_ball_any_only' if any(ball) else 'timeout_before_ball'
    return 'non_success_other'


def decision(control, treatment, *, complete, finite, integrity_ok):
    """The frozen integer advancement rule; never adjudicate a partial study."""
    if not (complete and finite and integrity_ok):
        return None
    if len(control) != 3 or len(treatment) != 3 or any(
        type(x) is not int or not 0 <= x <= MAPS for x in [*control, *treatment]
    ):
        raise ValueError('decision requires three paired integer success counts in [0,12]')
    differences = [b - c for c, b in zip(control, treatment)]
    rule = {
        'sum_integer_differences_at_least_3': sum(differences) >= 3,
        'at_least_two_positive_seed_differences': sum(d > 0 for d in differences) >= 2,
        'no_seed_loses_more_than_one_map': min(differences) >= -1,
        'all_six_complete_finite_and_integrity_checked': True,
    }
    advance = all(rule.values())
    if sum(control) == 0:
        label = ('PRELIMINARY_FIXED_BUDGET_LEARNING_GAIN' if advance
                 else 'INCONCLUSIVE_UNESTABLISHED_CAPTURE_FLOOR')
    else:
        label = 'ADVANCEMENT_RULE_MET' if advance else 'NEGATIVE_OR_MIXED_FIXED_BUDGET'
    return {
        'label': label,
        'advance': advance,
        'integer_differences_by_seed': dict(zip(map(str, SEEDS), differences)),
        'paired_fraction_differences_by_seed': dict(zip(map(str, SEEDS), [d / MAPS for d in differences])),
        'mean_paired_fraction_difference': sum(differences) / (len(SEEDS) * MAPS),
        'rule_checks': rule,
        'control_all_seed_gate_met': all(c >= 8 for c in control),
        'treatment_all_seed_gate_met': all(b >= 8 for b in treatment),
        'official_gate': 'at least 8/12 final successes in every training seed',
        'interpretation': 'Engineering advancement rule only; n=3 training seeds; not a significance test.',
        'automatic_launch_or_extension_authorized': False,
    }


def metrics_summary(path, final_complete, issues):
    path = Path(path)
    rows = []
    tail_ignored = False
    if path.exists():
        lines = path.read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('record is not an object')
                rows.append(row)
            except ValueError as exc:
                if not final_complete and index == len(lines) - 1:
                    tail_ignored = True  # append may be in progress; do not infer a failure.
                else:
                    issues.append(f'{path}:{index + 1}: invalid metric record: {exc}')
    progress = [r for r in rows if r.get('kind') == 'progress']
    losses = [r for r in progress if 'critic_loss' in r]
    nonfinite = []
    for row in losses:
        for key in LOSS_KEYS:
            if not finite_number(row.get(key)):
                nonfinite.append({'step': row.get('step'), 'key': key})
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                nonfinite.append({'step': row.get('step'), 'key': key})
    episodes = [r for r in rows if r.get('kind') == 'episode']
    for row in episodes:
        for key in ('return_', 'base_return', 'shaping_return'):
            if key in row and not finite_number(row[key]):
                nonfinite.append({'step': row.get('step'), 'key': key})
    if nonfinite:
        issues.append(f'{path}: nonfinite/missing learning diagnostics: {nonfinite[:8]}')
    latest = max(losses, key=lambda r: r.get('step', -1)) if losses else {}
    finite_evidence = bool(losses) and not nonfinite
    final_evidence = finite_evidence and latest.get('step') == FINAL_STEP and latest.get('updates') == 39000
    if final_complete and not final_evidence:
        issues.append(f'{path}: final finite progress at step=40000, updates=39000 is not established')
    values = lambda key: [float(r[key]) for r in losses if finite_number(r.get(key))]
    diagnostic = {}
    for key in ('critic_clipped', 'actor_clipped', 'entropy', 'alpha', 'td_abs',
                'critic_grad', 'actor_grad', 'terminal_sample_fraction'):
        samples = values(key)
        diagnostic[key] = {
            'progress_sample_mean': mean(samples),
            'latest': float(latest[key]) if finite_number(latest.get(key)) else None,
            'sample_count': len(samples),
        }
    completed_returns = {}
    for key in ('base_return', 'shaping_return', 'return_'):
        samples = [float(r[key]) for r in episodes if finite_number(r.get(key))]
        completed_returns[key] = {'completed_episode_total': sum(samples), 'episodes_with_field': len(samples)}
    return {
        'progress_records': len(progress), 'learning_progress_records': len(losses),
        'finite_diagnostics_observed': finite_evidence if losses else None,
        'finite_final_diagnostics_established': final_evidence,
        'finite_evidence_interpretation': 'finite_diagnostics_observed describes available learning progress (null means no loss sample yet). A false finite_final_diagnostics_established before final40k means pending final evidence, not a nonfinite failure.',
        'latest_progress_step': max((r.get('step', 0) for r in progress), default=None),
        'latest_update_count': latest.get('updates'),
        'incomplete_live_metrics_tail_ignored': tail_ignored,
        'diagnostics': diagnostic,
        'completed_episode_count': len(episodes),
        'completed_episode_returns': completed_returns,
        'return_scope': 'Only completed episodes logged in metrics; excludes the unfinished episode. Shaped returns are not comparable performance scores.',
        'diagnostic_scope': 'Diagnostics are sampled progress log values, not every training update.',
    }


def map_signature(rows, issues, context):
    if not isinstance(rows, list) or len(rows) != MAPS:
        issues.append(f'{context}: expected exactly 12 map records')
        return None
    signature = {}
    for row in rows:
        if not isinstance(row, dict):
            issues.append(f'{context}: map record is not an object')
            return None
        seed, family, scene = row.get('map_seed'), row.get('family'), row.get('scene_sha256')
        if type(seed) is not int or not isinstance(family, str) or not family or not isinstance(scene, str) or not scene:
            issues.append(f'{context}: missing map seed/family/scene hash')
            return None
        if seed in signature:
            issues.append(f'{context}: duplicate map seed {seed}')
            return None
        if type(row.get('success')) is not bool:
            issues.append(f'{context}: nonboolean success for map {seed}')
            return None
        signature[seed] = (family, scene)
    return signature


def trajectory_secondary(directory, step, row, warnings):
    distances = row.get('final_goal_m_by_uav')
    worst_distance = (max(distances) if isinstance(distances, list) and len(distances) == 3
                      and all(finite_number(x) for x in distances) else None)
    worst_speed = None
    path = directory / f'eval_{step}_{row["map_seed"]}.npz'
    if path.exists():
        try:
            import numpy as np
            with np.load(path, allow_pickle=False) as trace:
                velocity = trace['velocities']
                if velocity.ndim != 3 or velocity.shape[1:] != (3, 3) or not len(velocity):
                    raise ValueError('expected velocities (T,3,3)')
                speeds = np.linalg.norm(velocity[-1], axis=-1)
                if not np.isfinite(speeds).all():
                    raise ValueError('nonfinite last velocity')
                worst_speed = float(speeds.max())
                if worst_distance is None and 'positions' in trace and 'goals' in trace:
                    d = np.linalg.norm(trace['positions'][-1] - trace['goals'], axis=-1)
                    if d.shape == (3,) and np.isfinite(d).all():
                        worst_distance = float(d.max())
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f'{path}: secondary trajectory unavailable: {exc}')
    return {'map_seed': row['map_seed'], 'worst_agent_goal_distance_m': worst_distance,
            'worst_agent_speed_mps': worst_speed}


def evaluation_summary(rows, step, directory=None, warnings=None):
    balls = [r.get('ever_in_ball_by_uav') for r in rows]
    ball_available = all(isinstance(b, list) and len(b) == 3 and all(type(x) is bool for x in b) for b in balls)
    occupancy = [r.get('max_simultaneous_capture_occupancy') for r in rows]
    occupancy_available = all(type(v) is int and 0 <= v <= 3 for v in occupancy)
    result = {
        'step': step, 'maps': len(rows), 'successes': sum(r['success'] for r in rows),
        'ever_ball_any_maps': sum(any(b) for b in balls) if ball_available else None,
        'ever_ball_all_maps': sum(all(b) for b in balls) if ball_available else None,
        'exclusive_failure_reasons': dict(Counter(failure_reason(r) for r in rows if not r['success'])),
        'collision_flags_nonexclusive': {key: sum(bool(r.get(key)) for r in rows) for key in COLLISIONS},
        'max_simultaneous_capture_occupancy_by_map': dict(zip((str(r['map_seed']) for r in rows), occupancy)),
        'mean_max_simultaneous_capture_occupancy': mean(occupancy) if occupancy_available else None,
    }
    if directory is not None:
        secondary = [trajectory_secondary(directory, step, r, warnings if warnings is not None else []) for r in rows]
        speeds = [r['worst_agent_speed_mps'] for r in secondary]
        distances = [r['worst_agent_goal_distance_m'] for r in secondary]
        result['secondary'] = {
            'by_map': secondary,
            'speed_maps_available': sum(x is not None for x in speeds),
            'mean_worst_agent_speed_mps': mean(speeds) if all(x is not None for x in speeds) else None,
            'mean_worst_agent_goal_distance_m': mean(distances) if all(x is not None for x in distances) else None,
            'scope': 'Last stored velocity only; no speed inferred from success/ever-ball flags. Missing traces remain null.',
        }
    return result


def read_history(root, seed, root_completion, issues, warnings):
    directory = root / f'maneuver_static_{seed}'
    completion = optional_json(directory / 'completion.json', issues)
    complete = bool(isinstance(completion, dict) and completion.get('seed') == seed and completion.get('step') == FINAL_STEP)
    metrics = metrics_summary(directory / 'metrics.jsonl', complete, issues)
    evaluations, signatures = {}, {}
    if directory.exists():
        for path in sorted(directory.glob('evaluation_*.json')):
            match = EVAL_NAME.fullmatch(path.name)
            if not match:
                continue
            step = int(match.group(1))
            rows = optional_json(path, issues)
            signature = map_signature(rows, issues, str(path))
            if signature is None:
                continue
            if any(row.get('step') != step for row in rows):
                issues.append(f'{path}: map record step differs from filename')
                continue
            signatures[step] = signature
            evaluations[step] = evaluation_summary(rows, step)
            if step == FINAL_STEP:
                evaluations[step] = evaluation_summary(rows, step, directory, warnings)
    latest_step = max(evaluations, default=None)
    final = evaluations.get(FINAL_STEP)
    if complete and final is None:
        issues.append(f'{directory}: completion exists but valid final 40000 evaluation is missing')
    if complete and final is not None and 'final_development_success' in completion:
        saved = completion['final_development_success']
        if not finite_number(saved) or abs(saved - final['successes'] / MAPS) > 1e-12:
            issues.append(f'{directory}: completion success differs from final evaluation')
    return {
        'seed': seed, 'directory': str(directory), 'completed_40000': complete,
        'root_complete': bool(isinstance(root_completion, dict) and root_completion.get('complete') is True),
        'latest_evaluation': evaluations.get(latest_step), 'final_evaluation': final,
        'curves': [evaluations[s] for s in sorted(evaluations)],
        'metrics': metrics,
    }, signatures


def check_manifests(manifests, issues):
    control, treatment = manifests['control'], manifests['treatment']
    if not isinstance(control, dict) or not isinstance(treatment, dict):
        issues.append('Both root run_manifest.json files are required')
        return False
    sources = control.get('sources')
    source_match = bool(isinstance(sources, dict) and sources and sources == treatment.get('sources'))
    if not source_match:
        issues.append('Root source-hash dictionaries are missing or differ')
    expected = {'steps_per_seed': FINAL_STEP, 'warmup': 1000, 'replay_capacity': 100000,
                'probe_mode': 'static', 'set_mode': 'pool', 'device': 'cuda', 'smoke': False}
    for arm, manifest in manifests.items():
        for key, value in expected.items():
            if manifest.get(key) != value:
                issues.append(f'{arm} manifest {key} must equal {value!r}')
        if manifest.get('seed_list') != list(SEEDS):
            issues.append(f'{arm} manifest seed_list does not match the frozen three seeds')
        capture = manifest.get('capture_train_and_gate')
        if capture != {'ball_m': 4.0, 'speed_mps': 2.0, 'latch': False}:
            issues.append(f'{arm} manifest capture contract is not frozen 4/2 latch=False')
        sync = manifest.get('sync_capture', {})
        expected_sync = {'enabled': arm == 'treatment', 'coefficient': .5, 'temperature': .1,
                         'ball_m': 4.0, 'speed_mps': 2.0}
        if sync != expected_sync:
            issues.append(f'{arm} manifest sync_capture differs from the frozen B-off/B-on contract')
    for key in ('development_maps_used', 'reshape', 'probe_config', 'parameter_counts'):
        if not control.get(key) or control.get(key) != treatment.get(key):
            issues.append(f'Paired manifest field {key} is missing or differs')
    return source_match


def curve_diagnostics(control, treatment):
    paired = []
    for c, b in zip(control, treatment):
        cc = {r['step']: r for r in c['curves']}
        bb = {r['step']: r for r in b['curves']}
        common = sorted(set(cc) & set(bb))
        row = {'seed': c['seed'], 'common_evaluation_steps': common, 'grid_starts_at_step_zero': False}
        for arm, values in (('control', cc), ('treatment', bb)):
            scores = [values[s]['successes'] / MAPS for s in common]
            area = sum((common[i] - common[i - 1]) * (scores[i] + scores[i - 1]) / 2
                       for i in range(1, len(common))) if len(common) >= 2 else None
            row[arm] = {
                'success_fraction_auc_step_units_on_common_observed_grid': area,
                'normalized_auc_on_observed_interval': area / (common[-1] - common[0]) if area is not None else None,
                'peak_to_last_common_drop': max(scores) - scores[-1] if scores else None,
                'last_common_is_final_40000': bool(common and common[-1] == FINAL_STEP),
            }
        paired.append(row)
    return paired


def analyze_study(control, treatment):
    roots = {'control': Path(control).resolve(), 'treatment': Path(treatment).resolve()}
    issues, warnings = [], []
    manifests = {arm: optional_json(root / 'run_manifest.json', issues) for arm, root in roots.items()}
    source_match = check_manifests(manifests, issues)
    completions = {arm: optional_json(root / 'completion.json', issues) for arm, root in roots.items()}
    arms, signatures = {}, {}
    for arm, root in roots.items():
        arms[arm], signatures[arm] = [], {}
        for seed in SEEDS:
            history, signature = read_history(root, seed, completions[arm], issues, warnings)
            arms[arm].append(history)
            signatures[arm][seed] = signature
    root_complete = all(isinstance(c, dict) and c.get('complete') is True and c.get('smoke') is False
                        for c in completions.values())
    complete = root_complete and all(r['completed_40000'] for rows in arms.values() for r in rows)
    final_map_match = None
    if complete:
        final_map_match = True
        canonical = None
        for arm in roots:
            completion = completions[arm]
            runs = completion.get('runs', [])
            if len(runs) != 3 or {r.get('seed') for r in runs} != set(SEEDS) or any(r.get('step') != FINAL_STEP for r in runs):
                issues.append(f'{arm}: root completion does not contain exactly the three frozen 40000 histories')
            manifest = manifests[arm] or {}
            expected_maps = manifest.get('development_maps_used', [])
            expected_pairs = {(r.get('seed'), r.get('family')) for r in expected_maps}
            for seed in SEEDS:
                signature = signatures[arm][seed].get(FINAL_STEP)
                if signature is None or len(expected_pairs) != MAPS or {(s, v[0]) for s, v in signature.items()} != expected_pairs:
                    final_map_match = False
                if canonical is None and signature is not None:
                    canonical = signature
                if signature != canonical:
                    final_map_match = False
        if not final_map_match:
            issues.append('Final seed/family/scene hashes are missing, inconsistent across histories, or differ from manifest maps')
    finite = all(r['metrics']['finite_final_diagnostics_established'] for rows in arms.values() for r in rows)
    ready = complete and finite and not issues
    primary = decision(
        [r['final_evaluation']['successes'] for r in arms['control']] if ready else [],
        [r['final_evaluation']['successes'] for r in arms['treatment']] if ready else [],
        complete=complete, finite=finite, integrity_ok=not issues,
    )
    return {
        'analysis_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'COMPLETE' if complete else 'IN_PROGRESS',
        'decision_ready': ready, 'decision': primary,
        'control_root': str(roots['control']), 'treatment_root': str(roots['treatment']),
        'source_hashes_match': source_match, 'final_map_pairing_verified': final_map_match,
        'all_six_finite_final_diagnostics_established': finite,
        'integrity_issues': issues, 'secondary_warnings': warnings,
        'arms': arms, 'paired_curve_diagnostics': curve_diagnostics(arms['control'], arms['treatment']),
        'interpretation': 'Three independent training seeds; 12 development maps are repeated measurements, not 36 independent training trials. No significance, safety, predictive-content or novelty claim. No sealed test or weights loaded.',
        'partial_policy': 'IN_PROGRESS reports latest observations and curves only; it does not apply or forecast the advancement decision.',
        'validation_scope': 'Root source hashes, recorded manifest contracts, completion markers, sampled learning diagnostics and final development map metadata; model weights are not read.',
    }


def markdown_report(report):
    lines = ['# Static B 40k development study', '', f'**{report["status"]}**', '', report['interpretation'], '']
    if report['decision'] is None:
        lines += ['No advancement decision is calculated. All six final completions, finite diagnostics and integrity checks are required.', '']
    else:
        d = report['decision']
        lines += [f'**{d["label"]}**; engineering rule met: **{d["advance"]}**.', '',
                  f'Mean paired success-fraction difference: {d["mean_paired_fraction_difference"]:.6f}.', '',
                  '| Training seed | Control /12 | B /12 | B − control maps |', '|---|---:|---:|---:|']
        for c, b in zip(report['arms']['control'], report['arms']['treatment']):
            lines.append(f'| {c["seed"]} | {c["final_evaluation"]["successes"]} | {b["final_evaluation"]["successes"]} | {b["final_evaluation"]["successes"] - c["final_evaluation"]["successes"]:+d} |')
        lines += ['', 'This rule supports planning confirmation only. It launches no training, extension or sealed test.', '']
    if report['integrity_issues']:
        lines += ['## Integrity issues', ''] + [f'- {issue}' for issue in report['integrity_issues']] + ['']
    lines += ['## Latest observed development evaluations', '',
              '| Arm | Seed | Step | Success /12 | Any ball /12 | All ball /12 | Mean max occupancy |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm, histories in report['arms'].items():
        for h in histories:
            e = h['latest_evaluation'] or {}
            lines.append(f'| {arm} | {h["seed"]} | {e.get("step")} | {e.get("successes")} | {e.get("ever_ball_any_maps")} | {e.get("ever_ball_all_maps")} | {e.get("mean_max_simultaneous_capture_occupancy")} |')
    lines += ['', '## Sampled learning diagnostics and completed-episode reward scale', '',
              'Clip fractions and entropy/alpha below are sampled progress logs, not all updates. “pending final40k” means the final evidence is not due yet; it does not indicate nonfinite diagnostics. JSON finite_diagnostics_observed separately reports whether available learning samples are finite (null means no loss sample yet). Reward totals include completed episodes only; shaped training return is not a common cross-arm performance metric.', '',
              '| Arm | Seed | Final 40k finite evidence | Critic clip fraction | Latest entropy | Latest alpha | Completed episodes | Base total | Shaping total |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for arm, histories in report['arms'].items():
        for h in histories:
            m = h['metrics']
            d = m['diagnostics']
            r = m['completed_episode_returns']
            final_evidence = ('pending final40k' if not h['completed_40000'] else
                              'verified' if m['finite_final_diagnostics_established'] else 'NOT ESTABLISHED')
            lines.append(f'| {arm} | {h["seed"]} | {final_evidence} | {d["critic_clipped"]["progress_sample_mean"]} | {d["entropy"]["latest"]} | {d["alpha"]["latest"]} | {m["completed_episode_count"]} | {r["base_return"]["completed_episode_total"]} | {r["shaping_return"]["completed_episode_total"]} |')
    lines += ['', 'Per-map exclusive failures, nonexclusive collision flags, occupancy, final distance/speed availability, common-grid AUC and peak-to-last drops are retained in analysis.json. Missing trajectory velocity is null.', '',
              '![Development curves](curves.png)', '']
    return '\n'.join(lines)


def plot_curves(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    colors = {'control': '#2563eb', 'treatment': '#dc2626'}
    styles = ('-', '--', ':')
    for arm, histories in report['arms'].items():
        for index, history in enumerate(histories):
            curve = history['curves']
            for ax, key in zip(axes, ('successes', 'ever_ball_all_maps')):
                points = [(r['step'] / 1000, r[key]) for r in curve if r.get(key) is not None]
                if points:
                    ax.plot(*zip(*points), color=colors[arm], linestyle=styles[index],
                            label=f'{arm} seed {history["seed"]}', linewidth=1.5)
    for ax, title in zip(axes, ('Joint successes', 'All-agent eventual ball entry')):
        ax.set_title(title)
        ax.set_xlabel('Environment steps (thousands)')
        ax.set_ylim(-.3, 12.3)
        ax.set_yticks((0, 3, 6, 9, 12))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Development maps out of 12')
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle(f'{report["status"]} — descriptive curves; final 40k is the efficacy endpoint')
    fig.tight_layout(rect=(0, .12 if handles else 0, 1, .94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_artifacts(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'analysis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (directory / 'analysis.md').write_text(markdown_report(report), encoding='utf-8')
    plot_curves(report, directory / 'curves.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = analyze_study(args.control, args.treatment)
    write_artifacts(report, args.out)
    print(f'ANALYSIS status={report["status"]} decision_ready={report["decision_ready"]} '
          f'decision={report["decision"]["label"] if report["decision"] else None}', flush=True)
    for issue in report['integrity_issues']:
        print(f'INTEGRITY {issue}', flush=True)
    print(f'ARTIFACTS {args.out.resolve() / "analysis.json"} '
          f'{args.out.resolve() / "analysis.md"} {args.out.resolve() / "curves.png"}', flush=True)


if __name__ == '__main__':
    main()
