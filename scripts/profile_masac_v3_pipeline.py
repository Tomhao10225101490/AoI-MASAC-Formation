#!/usr/bin/env python3
"""Separate observation construction and Actor latency; no optimizer updates.

This is a local engineering profile. Other host workloads are not controlled.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from uav_planning.structured import StructuredMission
from uav_planning.shared_masac import Actor


def stats(values):
    return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                mean_ms=float(np.mean(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite a profile')
    torch.set_num_threads(1)
    torch.manual_seed(913001)
    actor = Actor().eval()
    records = []
    with torch.no_grad():
        for n in (3, 6):
            for mode in ('standard', 'distance', 'prediction', 'maneuver'):
                env = StructuredMission(n=n, mode=mode)
                env.reset(913001)
                env.v[:, 0] = 8.
                observation_ms, actor_ms, total_ms = [], [], []
                for i in range(args.samples + 10):
                    begin = time.perf_counter()
                    obs = env.observe()
                    encoded = torch.from_numpy(obs).unsqueeze(0)
                    ready = time.perf_counter()
                    actor(encoded, deterministic=True)
                    end = time.perf_counter()
                    if i >= 10:
                        observation_ms.append(1000 * (ready - begin))
                        actor_ms.append(1000 * (end - ready))
                        total_ms.append(1000 * (end - begin))
                records.append(dict(n=n, mode=mode, samples=args.samples,
                                    observation=stats(observation_ms), actor=stats(actor_ms),
                                    total=stats(total_ms), raw_ms=dict(
                                        observation=observation_ms, actor=actor_ms, total=total_ms)))
    result = dict(
        device='cpu', threads=1, platform=platform.platform(), torch=torch.__version__,
        weight_state='initialized Actor, fixed across all timing cases; no learned result',
        scenario='urban seed 913001; stationary positions, vx=8m/s; 10 warmup passes',
        limitations=['Local engineering profile while the budget-extension training is running.',
                     'No GPU or end-to-end simulator timing; no reserved-host latency guarantee.',
                     'Previous evaluation inference_ms measures Actor only, omitting observation/probe work.'],
        source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in ('uav_planning/structured.py', 'uav_planning/shared_masac.py',
                                    'scripts/profile_masac_v3_pipeline.py')},
        cases=records,
    )
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != 'raw_ms'} for r in records], indent=2))


if __name__ == '__main__':
    main()
