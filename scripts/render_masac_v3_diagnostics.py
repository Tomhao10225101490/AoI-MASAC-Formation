#!/usr/bin/env python3
"""Plot mechanism counterfactuals and real failed V3 evaluation trajectories."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/masac_v3_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def building(ax, lo, hi):
    x, y, z = lo
    X, Y, Z = hi
    v = np.array([[x,y,z], [X,y,z], [X,Y,z], [x,Y,z],
                  [x,y,Z], [X,y,Z], [X,Y,Z], [x,Y,Z]])
    faces = [[v[i] for i in ids] for ids in ((0,1,2,3), (4,5,6,7),
             (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7))]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolor=plt.cm.Blues(.2+.5*Z/90), alpha=.3,
        edgecolor='#718ba1', linewidth=.3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnosis', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    out = args.diagnosis.parent
    data = json.loads(args.diagnosis.read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#2375a8', '#df9338', '#218875']
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for row, color in zip(data['paired_probe'], colors):
        axs[0].plot(range(19), row['margins_m'], 'o-', ms=4,
                    label=f'a_max = {row["amax"]:.0f} m/s²', color=color)
    axs[0].axhline(0, color='#555555', ls='--', lw=1)
    axs[0].set(xlabel='Fixed acceleration probe index', ylabel='Minimum predicted clearance (m)',
               title='Direction-resolved control reserve', xticks=[0,3,6,9,12,15,18])
    axs[0].legend(fontsize=9)
    x = np.arange(3)
    for offset, label, key, color in (
        (-.25, 'Current clearance', 'current_clearance_m', '#a4b4c3'),
        (0, 'Constant-velocity preview', 'cv_preview_margin_m', '#576779'),
        (.25, 'Best tested control', 'best_tested_margin_m', '#218875'),
    ):
        values = [r[key] for r in data['paired_probe']]
        bars = axs[1].bar(x+offset, values, .24, label=label, color=color)
        for bar, value in zip(bars, values):
            axs[1].text(bar.get_x()+bar.get_width()/2,
                        value+(.3 if value >= 0 else -.3), f'{value:+.2f}',
                        ha='center', va='bottom' if value >= 0 else 'top', fontsize=8)
    axs[1].axhline(0, color='#555555', lw=1)
    axs[1].set(xticks=x, xticklabels=['2 m/s²', '4 m/s²', '6 m/s²'], ylim=(-11,18),
               xlabel='Maximum acceleration', ylabel='Clearance (m)',
               title='Same position, speed and geometry')
    axs[1].legend(fontsize=8, loc='upper center', ncol=1)
    for ax in axs:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Mechanism signal check | 9 m clearance, 8 m/s closing speed, 2 s preview', fontsize=12)
    fig.text(.5, .02, 'Model-based counterfactual only. No learned-policy improvement or safety guarantee.',
             ha='center', color='#555555', fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    fig.savefig(out / 'mechanism_probe.png', dpi=190)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    sources = []
    directory = args.parent / 'standard_411'
    records = json.loads((directory / 'evaluation_20000.json').read_text())
    for col, record in enumerate(records[:3]):
        seed = record['map_seed']
        file = directory / f'eval_20000_{seed}.npz'
        with np.load(file) as episode:
            pos, goals, boxes, dynamic = [episode[k] for k in ('positions', 'goals', 'buildings', 'dynamic')]
        sources.append(dict(path=str(file.resolve()), sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                            train_seed=411, step=20000, map_seed=seed, result=record))
        ax = fig.add_subplot(2, 3, col+1, projection='3d')
        top = fig.add_subplot(2, 3, col+4)
        for lo, hi in boxes:
            building(ax, lo, hi)
            top.add_patch(Rectangle(lo[:2], *(hi-lo)[:2], facecolor=plt.cm.Blues(.2+.5*hi[2]/90),
                                    edgecolor='white', lw=.6, alpha=.8))
        for i, color in enumerate(colors):
            ax.plot(*pos[:,i].T, color=color, lw=2.2)
            ax.scatter(*pos[0,i], color=color, s=22)
            ax.scatter(*goals[i], color=color, marker='*', s=80)
            ax.scatter(*pos[-1,i], color=color, marker='x', s=38)
            top.plot(*pos[:,i,:2].T, color=color, lw=2.2, label=f'UAV {i+1}')
            top.scatter(*pos[0,i,:2], color=color, s=23)
            top.scatter(*goals[i,:2], color=color, marker='*', s=80)
            top.scatter(*pos[-1,i,:2], color=color, marker='x', s=38)
        for i in range(dynamic.shape[1]):
            ax.plot(*dynamic[:,i].T, color='#c85058', alpha=.65, lw=.8)
            top.plot(*dynamic[:,i,:2].T, color='#c85058', alpha=.6, lw=.8)
        failures = [key.removesuffix('_collision') for key in ('static_collision', 'boundary_collision',
                    'uav_collision', 'dynamic_collision') if record[key]]
        ax.set(xlim=(0,240), ylim=(0,240), zlim=(0,90), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)',
               title=f'{record["layout"].capitalize()} | failure: {", ".join(failures)}\n'
                     f't = {record["time"]:.1f} s; map {seed}')
        ax.set_box_aspect((240,240,90))
        ax.set_proj_type('ortho')
        ax.view_init(32,-62)
        ax.set_xticks([0,120,240]); ax.set_yticks([0,120,240]); ax.set_zticks([0,45,90])
        top.set(xlim=(0,240), ylim=(0,240), xlabel='x (m)', ylabel='y (m)')
        top.set_aspect('equal')
        top.grid(alpha=.1)
        if col == 0:
            top.legend(fontsize=8, loc='upper center', ncol=3)
    fig.suptitle('Actual Standard MASAC V3 evaluation paths | training seed 411, 20,000 steps\n'
                 'All three episodes failed. Circles: starts; stars: goals; crosses: last recorded positions.', fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    fig.savefig(out / 'actual_failed_trajectories.png', dpi=180)
    plt.close(fig)
    (out / 'figure_sources.json').write_text(json.dumps(sources, indent=2))
    print(json.dumps(dict(figures=[str(out/'mechanism_probe.png'), str(out/'actual_failed_trajectories.png')],
                          trajectories=len(sources))))


if __name__ == '__main__':
    main()
