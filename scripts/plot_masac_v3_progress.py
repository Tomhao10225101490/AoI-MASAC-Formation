#!/usr/bin/env python3
"""Plot completed records, including an explicitly declared V3 continuation parent."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/masac_v3_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rolling(values, window=20):
    return [np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = json.loads((root / 'protocol.json').read_text())
    roots = [root]
    if cfg.get('parent'):
        parent = Path(cfg['parent'])
        parent_cfg = json.loads((parent / 'protocol.json').read_text())
        if parent_cfg['version'] != cfg['version']:
            raise ValueError('Cannot combine different algorithm versions')
        roots.insert(0, parent)
    names = sorted({d.name for base in roots for d in base.iterdir()
                    if d.is_dir() and (d / 'training.jsonl').exists()})
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
    records = []
    for name in names:
        progress, episodes, evaluations = {}, {}, {}
        for base in roots:
            path = base / name / 'training.jsonl'
            if path.exists():
                for line in path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get('kind') == 'progress':
                        progress[row['step']] = row
                    elif row.get('kind') == 'episode':
                        episodes[row['step']] = row
            for file in (base / name).glob('evaluation_*.json'):
                data = json.loads(file.read_text())
                if data:
                    evaluations[data[0]['step']] = dict(
                        step=data[0]['step'], successes=sum(x['success'] for x in data),
                        maps=len(data), mean_remaining_distance_m=float(np.mean(
                            [x['remaining_distance'] for x in data])))
        progress = [progress[k] for k in sorted(progress)]
        episodes = [episodes[k] for k in sorted(episodes)]
        evaluations = [evaluations[k] for k in sorted(evaluations)]
        for ax, metric in ((axs[0, 0], 'critic_loss'), (axs[0, 1], 'actor_loss')):
            if progress:
                ax.plot([x['step'] for x in progress], rolling(
                    [x.get(metric, np.nan) for x in progress], 10), label=name)
        if episodes:
            steps = [r['step'] for r in episodes]
            axs[0, 2].plot(steps, rolling([r['return_'] for r in episodes]), label=name)
            axs[1, 0].plot(steps, rolling([r['time'] for r in episodes]), label=name)
        if evaluations:
            steps = [r['step'] for r in evaluations]
            axs[1, 1].plot(steps, [r['mean_remaining_distance_m'] for r in evaluations], 'o-', ms=3, label=name)
            axs[1, 2].plot(steps, [r['successes']/r['maps'] for r in evaluations], 'o-', ms=3, label=name)
        records.append(dict(run=name, last_training_step=progress[-1]['step'] if progress else 0,
                            complete=(root / name / 'audit.json').exists(), evaluations=evaluations))
    titles = ['Twin critic loss (10-record mean)', 'Actor objective (10-record mean)',
              'Episode return (20-episode mean)', 'Episode duration (seconds, 20-episode mean)',
              'Validation mean distance to goals (m)', 'Validation all-agent success']
    for ax, title in zip(axs.flat, titles):
        ax.set(xlabel='Total environment steps per seed', ylabel=title)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=8)
        if cfg.get('resume_step'):
            ax.axvline(cfg['resume_step'], color='#777777', ls=':', lw=1)
    axs[1, 2].set_ylim(-.02, 1.02)
    phase = 'complete' if (root / 'completion.json').exists() else 'in progress'
    fig.suptitle(f'Standard MASAC V3 | development budget diagnosis | {phase}\n'
                 'Repeated validation maps are development data; seeds are separate training runs.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(root / 'progress.png', dpi=180)
    (root / 'progress_plot_snapshot.json').write_text(json.dumps(dict(
        roots=[str(p) for p in roots], runs=records, status=phase), indent=2))
    print(json.dumps([{k: v for k, v in r.items() if k != 'evaluations'} for r in records]))


if __name__ == '__main__':
    main()
