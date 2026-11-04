#!/usr/bin/env python3
"""Read-only byte/metadata audit of a registered 40k control/B seed pair.

Uses only the standard library. checkpoint.pt and replay.npz are streamed as
opaque bytes, never deserialized. This is not an exact CUDA resume test, a
policy evaluation, or a study-level efficacy/completion decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1511, 1522, 1533)
STEP = 40000
UPDATES = 39000
FIELDS = ('actor_loss', 'critic_loss', 'alpha', 'actor_grad', 'critic_grad',
          'actor_grad_post', 'critic_grad_post', 'q_mean', 'target_mean', 'td_abs', 'entropy')
CONFIGS = {
    'control': 'configs/masac_v4/sync_capture_40k_control_cuda_20261105.json',
    'b': 'configs/masac_v4/sync_capture_40k_b_cuda_20261105.json',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def repository_file(name):
    """Resolve symlinks/.. before requiring the source/config to stay in repo."""
    path = (ROOT / Path(name)).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f'Source/config path escapes repository: {name}') from exc
    if not path.is_file():
        raise ValueError(f'Source/config file is missing: {path}')
    return path


def file_digest(path, expected):
    h = hashlib.sha256()
    byte_count = 0
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
            byte_count += len(chunk)
    observed = h.hexdigest()
    return {'path': str(Path(path).resolve()), 'bytes': byte_count, 'sha256': observed,
            'expected_sha256': expected, 'matches': observed == expected}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check(record, name, value):
    record['checks'][name] = bool(value)
    print(f'CHECK arm={record["arm"]} seed={record["seed"]} '
          f'{name}={"PASS" if value else "FAIL"}', flush=True)


def verify_arm(record, root):
    arm, seed = record['arm'], record['seed']
    root = Path(root).resolve()
    directory = root / f'maneuver_static_{seed}'
    checkpoint = directory / 'checkpoint_040000'
    record.update(root=str(root), directory=str(directory), checkpoint=str(checkpoint))
    manifest = read(root / 'run_manifest.json')
    saved = read(checkpoint / 'manifest.json')
    completion = read(directory / 'completion.json')
    rows = read(directory / 'evaluation_40000.json')
    metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
               if line.strip()]
    progress = [row for row in metrics if isinstance(row, dict) and row.get('kind') == 'progress']
    trained = [row for row in progress if finite(row.get('updates')) and row['updates'] > 0]
    latest = progress[-1] if progress else {}
    finite_logged = bool(trained) and all(all(finite(row.get(key)) for key in FIELDS) for row in trained)
    record.update(
        completion=completion,
        updates=latest.get('updates'),
        logged_final_step=latest.get('step'),
        logged_last_eval_step=latest.get('last_eval_step'),
        logged_trained_progress_rows=len(trained),
        finite_logged_diagnostics=finite_logged,
        checked_diagnostic_fields=list(FIELDS),
        diagnostic_scope='Only the logged post-warmup progress samples are checked. The frozen schedule normally yields 195 such samples; these are not 39000 individually recorded update diagnostics.',
        last_eval_scope='The final progress log is written before final evaluation, so last_eval_step=38000 is valid. evaluation_40000.json and seed completion are authoritative for the final endpoint.',
    )
    check(record, 'seed_completion_matches_40000',
          completion.get('seed') == seed and completion.get('step') == STEP)
    valid_rows = (isinstance(rows, list) and len(rows) == 12 and
                  all(isinstance(row, dict) and row.get('step') == STEP and
                      type(row.get('success')) is bool for row in rows))
    check(record, 'twelve_final40000_map_records', valid_rows)
    success = sum(row['success'] for row in rows) if valid_rows else None
    record['success_maps'] = success
    fraction = completion.get('final_development_success')
    check(record, 'seed_completion_success_matches_final_eval',
          success is not None and finite(fraction) and abs(fraction - success / 12) < 1e-12)
    check(record, 'progress_records_have_valid_steps_and_updates', bool(progress) and all(
        type(row.get('step')) is int and 0 <= row['step'] <= STEP and
        type(row.get('updates')) is int and 0 <= row['updates'] <= UPDATES for row in progress))
    check(record, 'final_progress_40000_with_39000_updates',
          latest.get('step') == STEP and latest.get('updates') == UPDATES)
    check(record, 'finite_logged_postwarmup_diagnostics', finite_logged)
    check(record, 'checkpoint_manifest_complete_seed_and_step',
          saved.get('complete') is True and saved.get('step') == STEP and saved.get('seed') == seed)
    sources = manifest.get('sources')
    if not isinstance(sources, dict) or not sources or not all(
        isinstance(name, str) and isinstance(expected, str) for name, expected in sources.items()
    ):
        raise ValueError(f'{root}: run manifest has no valid nonempty source-hash dictionary')
    record['recorded_sources'] = sources
    check(record, 'snapshot_manifest_sources_match_run_manifest', saved.get('sources') == sources)
    record['source_files'] = {}
    for name, expected in sources.items():
        path = repository_file(name)
        item = file_digest(path, expected)
        record['source_files'][name] = item
        print(f'SOURCE_HASH arm={arm} source={name} bytes={item["bytes"]} '
              f'sha256={item["sha256"]} matches={item["matches"]}', flush=True)
    record['changed_sources'] = [name for name, item in record['source_files'].items() if not item['matches']]
    check(record, 'current_repository_source_bytes_match_run_manifest', not record['changed_sources'])
    config = repository_file(CONFIGS[arm])
    config_hash = manifest.get('config_hashes', {}).get('protocol')
    record['config'] = file_digest(config, config_hash)
    check(record, 'current_registered_config_matches_run_manifest', record['config']['matches'])
    record['binaries'] = {}
    for filename, key in (('checkpoint.pt', 'checkpoint_sha256'), ('replay.npz', 'replay_sha256')):
        item = file_digest(checkpoint / filename, saved.get(key))
        record['binaries'][filename] = item
        print(f'FILE_HASH arm={arm} seed={seed} path={item["path"]} bytes={item["bytes"]} '
              f'sha256={item["sha256"]} matches={item["matches"]}', flush=True)
    check(record, 'checkpoint_and_replay_bytes_match_snapshot_manifest',
          all(item['matches'] for item in record['binaries'].values()))
    # A worker may already be training the next seed. Status is informational.
    try:
        record['current_status_informational_only'] = read(root / 'status.json')
    except (OSError, ValueError) as exc:
        record['current_status_informational_only'] = {'read_error': str(exc)}
    record['seed_artifact_checks_passed'] = all(record['checks'].values())
    print(f'SEED_AUDIT arm={arm} seed={seed} final_success={success}/12 '
          f'updates={record["updates"]} logged_diagnostics={len(trained)} '
          f'finite_logged={finite_logged} checks_passed={record["seed_artifact_checks_passed"]}', flush=True)


def json_safe(value):
    """Retain a failed audit even if a malformed log contains NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_value': repr(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def verify_pair(control, treatment, seed):
    if seed not in SEEDS:
        raise ValueError(f'Seed must be one of the registered seeds: {SEEDS}')
    records = []
    for arm, root in (('control', control), ('b', treatment)):
        record = {'arm': arm, 'seed': seed, 'step': STEP, 'checks': {},
                  'seed_artifact_checks_passed': False}
        records.append(record)
        try:
            verify_arm(record, root)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            record['seed_artifact_checks_passed'] = False
            print(f'VERIFICATION_ERROR arm={arm} seed={seed} {record["error"]}', flush=True)
    source_pair_match = bool(records[0].get('recorded_sources')) and (
        records[0].get('recorded_sources') == records[1].get('recorded_sources'))
    passed = source_pair_match and all(r['seed_artifact_checks_passed'] for r in records)
    return {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if passed else 'FAIL', 'seed': seed, 'step': STEP,
        'repository': str(ROOT), 'source_hashes_match_between_arms': source_pair_match,
        'seed_pair_artifact_checks_passed': passed,
        'scope': 'Raw-byte per-seed artifact integrity, matching current repository source/config and logged finite diagnostics only. No torch import, weight/replay deserialization or policy execution.',
        'boundaries': [
            '195 expected logged diagnostics are sampled records, not 39000 individually logged update checks.',
            'Matching SHA256 bytes does not establish exact CUDA checkpoint/replay/optimizer/RNG restoration.',
            'Worker status may already refer to the next seed and is not a completion criterion.',
            'Final progress last_eval_step may be 38000; final40000 evaluation is checked separately.',
            'This checks one seed pair, not all six histories, the study advancement decision or sealed test outcomes.',
        ],
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--out', type=Path, required=True, help='output directory for verification.json')
    args = parser.parse_args()
    report = verify_pair(args.control, args.treatment, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / 'verification.json'
    output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'PAIR_SEED_VERIFICATION seed={args.seed} status={report["status"]} '
          f'source_hashes_match_between_arms={report["source_hashes_match_between_arms"]}', flush=True)
    print(f'ARTIFACT {output.resolve()}', flush=True)
    return 0 if report['seed_pair_artifact_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
