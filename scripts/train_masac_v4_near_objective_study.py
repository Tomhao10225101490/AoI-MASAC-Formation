#!/usr/bin/env python3
"""New fixed final40k near-objective study; no warm start or pilot continuation."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_masac_v4_maneuver as trainer
from uav_planning.v4_near_study_runtime import install_training_reward

PROTOCOL = 'docs/design/MASAC_V4_NEAR_OBJECTIVE_40K_PROTOCOL_20261105.md'
ANALYZER = 'scripts/analyze_near_objective_study.py'
COORDINATOR = 'scripts/launch_near_objective_study.py'
SEEDS = [1611, 1622, 1633]
RULE = dict(minimum_sum_success_gain=3, minimum_positive_seed_pairs=2,
            minimum_pair_success_gain=-1, absolute_success_count_per_seed=8)


def normalized(cfg):
    value = copy.deepcopy(cfg)
    for key in ('experiment_id', 'suite_experiment_id', 'change_set'):
        value.pop(key)
    value['near_objective']['mode'] = 'MATCHED_MODE_PLACEHOLDER'
    return value


def validate_pair(path, cfg):
    mode = cfg['near_objective']['mode']
    if mode not in ('original', 'potential') or cfg['near_objective'] != dict(
            mode=mode, coefficient=.4, radius_m=15., gamma=.99):
        raise ValueError('Unknown or modified near-objective intervention')
    required = dict(seeds=SEEDS, steps=40000, warmup=1000, batch_size=128,
                    replay_capacity=100000, eval_interval=2000, checkpoint_interval=10000,
                    log_interval=200, device='cuda', set_mode='pool', n_agents=3,
                    probe_modes=['static'], default_probe_mode='static',
                    preregistered_protocol=PROTOCOL, decision_rule=RULE,
                    execution=dict(layout='near-objective-independent-histories-v1', max_workers=4))
    if any(cfg.get(key) != value for key, value in required.items()) or cfg['sync_capture']['enabled']:
        raise ValueError('This entry freezes the registered two-arm, new-seed final40k study')
    legacy_config = json.loads((ROOT / 'configs/masac_v4/sync_capture_40k_control_cuda_20261105.json').read_text())
    for key in ('reshape', 'development_maps', 'families', 'sensor_range', 'gate', 'training_map_seed_offset',
                'training_seed_multiplier', 'sync_capture', 'compute_evidence'):
        if cfg[key] != legacy_config[key]:
            raise ValueError(f'Unexpected shared baseline change: {key}')
    if path.resolve() != (ROOT / f'configs/masac_v4/near_objective_40k_{mode}_cuda_20261105.json').resolve():
        raise ValueError('Use the named registered config, no substitute path')
    other_mode = 'potential' if mode == 'original' else 'original'
    other_path = ROOT / f'configs/masac_v4/near_objective_40k_{other_mode}_cuda_20261105.json'
    other = json.loads(other_path.read_text())
    if other['near_objective']['mode'] != other_mode or normalized(cfg) != normalized(other):
        raise ValueError('Paired arm configuration mismatch beyond the named formula/mode IDs')
    return {p.relative_to(ROOT).as_posix(): trainer.digest(p) for p in (path.resolve(), other_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--seed', type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    if args.torch_threads != 1:
        raise ValueError('Registered study freezes one Torch thread per CUDA worker')
    cfg = json.loads(args.protocol.read_text(encoding='utf-8'))
    pair_hashes = validate_pair(args.protocol, cfg)
    expected_out = (ROOT / 'results/masac_v4_near_objective_40k_sharded_cuda_20261105/runs' /
                    f'{cfg["near_objective"]["mode"]}_{args.seed}').resolve()
    if args.out.resolve() != expected_out:
        raise ValueError('Use the registered owned history shard output; no substitute attempts')
    legacy = json.loads((ROOT / 'results/masac_v4_sync_40k_control_cuda_20261105/run_manifest.json').read_text())
    if len(legacy['sources']) != 14 or any(trainer.digest(ROOT / p) != sha for p, sha in legacy['sources'].items()):
        raise ValueError('Original14 frozen sources changed; do not launch this study')
    trainer.SOURCES += ('uav_planning/v4_near_objective.py', 'uav_planning/v4_near_study_runtime.py',
                        'scripts/train_masac_v4_near_objective_study.py', PROTOCOL, ANALYZER, COORDINATOR)
    install_training_reward(trainer, cfg, args.out.resolve(), ROOT / PROTOCOL)
    original_write = trainer.write

    def write(path, obj):
        if Path(path).name == 'run_manifest.json':
            obj = dict(obj, paired_config_hashes=pair_hashes,
                       decision_analyzer=dict(path=ANALYZER, sha256=trainer.digest(ROOT / ANALYZER)),
                       study='new matched-budget near-objective final40k; no old continuation',
                       history_shard=dict(mode=cfg['near_objective']['mode'], training_seed=args.seed,
                                          layout='near-objective-independent-histories-v1'))
        original_write(path, obj)

    trainer.write = write
    print(f'NEAR-STUDY CONFIG={cfg} paired_config_hashes={pair_hashes} '
          'cold_start=true sealed_maps=false automatic_extension=false', flush=True)
    args.authorization_protocol = None
    args.probe_mode = 'static'
    args.smoke = False
    args.warm_start = None
    args.lock_path = None
    args.primary = False
    args.caffeinate_pid = None
    trainer.run(args)


if __name__ == '__main__':
    main()
