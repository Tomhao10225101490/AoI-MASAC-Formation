#!/usr/bin/env python3
"""Describe saved Replay goal coverage, not independent test success rates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite coverage evidence')
    cfg = json.loads((args.run / 'protocol.json').read_text())
    if cfg['version'] != 'structured-v3' or cfg['world'] != [240, 240, 90]:
        raise ValueError('This audit is specific to the V3 observation normalization')
    rows = []
    for file in sorted(args.run.glob('standard_*/replay_final.npz')):
        with np.load(file) as replay:
            observations = replay['obs']
            distance = np.linalg.norm(observations[..., :3] * 240., axis=-1)
            speed = np.linalg.norm(observations[..., 3:6] * cfg['vmax'], axis=-1)
            rows.append(dict(
                run=file.parent.name, transitions=len(distance),
                agent_states=int(distance.size), minimum_distance_m=float(distance.min()),
                distance_percentiles_m=dict(zip(('p01', 'p10', 'p50', 'p90'),
                                                np.percentile(distance, [1,10,50,90]).tolist())),
                agent_states_below_distance={str(d): int(np.count_nonzero(distance < d))
                                             for d in (4,10,30,50,100)},
                agent_states_with_goal_capture=int(np.count_nonzero((distance < 4) & (speed < 2))),
                replay_path=str(file.resolve()), replay_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            ))
    result = dict(
        evidence_level='training experience coverage; correlated states, not test episodes',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows,
        limitation='Pre-action Replay observations only; absence does not by itself identify the cause of learning failure.',
    )
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
