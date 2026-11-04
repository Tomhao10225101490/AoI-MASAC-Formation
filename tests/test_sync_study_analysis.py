import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'analyze_sync_study', Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_sync_study.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def adjudicate(control, treatment, **kwargs):
    return analysis.decision(control, treatment, complete=True, finite=True, integrity_ok=True, **kwargs)


def test_preregistered_positive_two_of_three_rule():
    result = adjudicate([3, 4, 5], [5, 6, 4])
    assert result['integer_differences_by_seed'] == {'1511': 2, '1522': 2, '1533': -1}
    assert result['mean_paired_fraction_difference'] == pytest.approx(1 / 12)
    assert result['advance'] is True


def test_mean_threshold_alone_with_only_one_positive_seed_does_not_pass():
    result = adjudicate([3, 4, 5], [6, 4, 5])
    assert result['rule_checks']['sum_integer_differences_at_least_3']
    assert not result['rule_checks']['at_least_two_positive_seed_differences']
    assert result['advance'] is False


def test_one_seed_losing_more_than_one_map_blocks_advancement():
    result = adjudicate([5, 5, 5], [8, 8, 3])
    assert result['rule_checks']['sum_integer_differences_at_least_3']
    assert result['rule_checks']['at_least_two_positive_seed_differences']
    assert not result['rule_checks']['no_seed_loses_more_than_one_map']
    assert not result['advance']


def test_both_zero_is_inconclusive_and_control_zero_signal_is_preliminary():
    result = adjudicate([0, 0, 0], [0, 0, 0])
    assert result['label'] == 'INCONCLUSIVE_UNESTABLISHED_CAPTURE_FLOOR'
    assert not result['advance']
    preliminary = adjudicate([0, 0, 0], [1, 2, 0])
    assert preliminary['label'] == 'PRELIMINARY_FIXED_BUDGET_LEARNING_GAIN'
    assert preliminary['advance']
    weak = adjudicate([0, 0, 0], [3, 0, 0])
    assert weak['label'] == 'INCONCLUSIVE_UNESTABLISHED_CAPTURE_FLOOR'


@pytest.mark.parametrize('missing', ['complete', 'finite', 'integrity_ok'])
def test_no_decision_without_all_evidence(missing):
    flags = dict(complete=True, finite=True, integrity_ok=True)
    flags[missing] = False
    assert analysis.decision([0, 0, 0], [12, 12, 12], **flags) is None


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def make_arm(root, counts, *, enabled, missing_seed=None):
    maps = [{'seed': 830001 + i, 'family': f'family-{i % 6}', 'level': 'L0'} for i in range(12)]
    manifest = {
        'sources': {'source.py': 'same-frozen-hash'}, 'seed_list': list(analysis.SEEDS),
        'steps_per_seed': 40000, 'warmup': 1000, 'replay_capacity': 100000,
        'probe_mode': 'static', 'set_mode': 'pool', 'device': 'cuda', 'smoke': False,
        'capture_train_and_gate': {'ball_m': 4., 'speed_mps': 2., 'latch': False},
        'sync_capture': {'enabled': enabled, 'coefficient': .5, 'temperature': .1, 'ball_m': 4., 'speed_mps': 2.},
        'development_maps_used': maps, 'reshape': {'capture_latch': False},
        'probe_config': {'horizon_s': 2.}, 'parameter_counts': {'total_trainable': 10},
    }
    dump(root / 'run_manifest.json', manifest)
    runs = []
    for seed, count in zip(analysis.SEEDS, counts):
        directory = root / f'maneuver_static_{seed}'
        completion = {'seed': seed, 'step': 40000, 'final_development_success': count / 12}
        if seed != missing_seed:
            dump(directory / 'completion.json', completion)
        runs.append(completion)
        rows = [{
            'map_seed': m['seed'], 'family': m['family'], 'scene_sha256': f'scene-{m["seed"]}',
            'step': 40000, 'success': i < count, 'timeout': i >= count,
            'ever_in_ball_by_uav': [i < count] * 3,
            'max_simultaneous_capture_occupancy': 3 if i < count else 0,
            'final_goal_m_by_uav': [1., 2., 3.],
        } for i, m in enumerate(maps)]
        dump(directory / 'evaluation_40000.json', rows)
        # This must never be mistaken for another map evaluation.
        dump(directory / 'evaluation_40000_summary.json', {'successes': 12})
        records = [
            {'kind': 'progress', 'step': 1000, 'updates': 0},
            {'kind': 'episode', 'step': 350, 'base_return': 2., 'shaping_return': -.01, 'return_': 1.99},
            {'kind': 'progress', 'step': 40000, 'updates': 39000, 'actor_loss': .1, 'critic_loss': 1.,
             'alpha': .05, 'entropy': -.5, 'critic_clipped': 1, 'actor_clipped': 0},
        ]
        (directory / 'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
    dump(root / 'completion.json', {'complete': True, 'smoke': False, 'runs': runs})


def test_complete_study_integrity_and_missing_velocity_remains_null(tmp_path):
    control, treatment = tmp_path / 'control', tmp_path / 'treatment'
    make_arm(control, [3, 4, 5], enabled=False)
    make_arm(treatment, [5, 6, 4], enabled=True)
    result = analysis.analyze_study(control, treatment)
    assert result['status'] == 'COMPLETE'
    assert result['integrity_issues'] == []
    assert result['decision']['advance']
    history = result['arms']['treatment'][0]
    assert len(history['curves']) == 1
    assert history['final_evaluation']['secondary']['mean_worst_agent_speed_mps'] is None
    assert history['metrics']['completed_episode_returns']['base_return']['completed_episode_total'] == 2.
    assert history['metrics']['diagnostics']['critic_clipped']['progress_sample_mean'] == 1.
    json.dumps(result, allow_nan=False)


def test_missing_history_never_adjudicates_even_with_high_latest_scores(tmp_path):
    control, treatment = tmp_path / 'control', tmp_path / 'treatment'
    make_arm(control, [0, 0, 0], enabled=False)
    make_arm(treatment, [12, 12, 12], enabled=True, missing_seed=1533)
    result = analysis.analyze_study(control, treatment)
    assert result['status'] == 'IN_PROGRESS'
    assert result['decision'] is None
    assert result['arms']['treatment'][2]['latest_evaluation']['successes'] == 12


@pytest.mark.parametrize('corruption', ['source', 'scene', 'family', 'nonfinite'])
def test_final_integrity_or_nonfinite_failure_blocks_primary_decision(tmp_path, corruption):
    control, treatment = tmp_path / 'control', tmp_path / 'treatment'
    make_arm(control, [3, 4, 5], enabled=False)
    make_arm(treatment, [5, 6, 4], enabled=True)
    directory = treatment / 'maneuver_static_1511'
    if corruption == 'source':
        path = treatment / 'run_manifest.json'
        data = json.loads(path.read_text())
        data['sources']['source.py'] = 'different'
        dump(path, data)
    elif corruption in ('scene', 'family'):
        path = directory / 'evaluation_40000.json'
        data = json.loads(path.read_text())
        data[0]['scene_sha256' if corruption == 'scene' else 'family'] = 'different'
        dump(path, data)
    else:
        path = directory / 'metrics.jsonl'
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'kind': 'progress', 'step': 40000, 'updates': 39000,
                                     'actor_loss': .1, 'critic_loss': float('nan'), 'alpha': .1, 'entropy': .1}) + '\n')
    result = analysis.analyze_study(control, treatment)
    assert result['status'] == 'COMPLETE'
    assert result['integrity_issues']
    assert result['decision'] is None


def test_saved_final_velocity_used_only_when_trace_is_available(tmp_path):
    import numpy as np
    control, treatment = tmp_path / 'control', tmp_path / 'treatment'
    make_arm(control, [3, 4, 5], enabled=False)
    make_arm(treatment, [5, 6, 4], enabled=True)
    velocity = np.zeros((2, 3, 3))
    velocity[-1, 1] = [3., 4., 0.]
    np.savez(treatment / 'maneuver_static_1511' / 'eval_40000_830001.npz', velocities=velocity)
    result = analysis.analyze_study(control, treatment)
    secondary = result['arms']['treatment'][0]['final_evaluation']['secondary']
    assert secondary['by_map'][0]['worst_agent_speed_mps'] == 5.
    assert secondary['by_map'][1]['worst_agent_speed_mps'] is None
    assert secondary['speed_maps_available'] == 1
    assert secondary['mean_worst_agent_speed_mps'] is None
