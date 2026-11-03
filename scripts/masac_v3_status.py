#!/usr/bin/env python3
"""Read-only live V3 viewer, including completion and stale heartbeat detection."""
import argparse
import datetime
import json
import time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def active_run():
    pointer = ROOT / 'results/masac_v3_active_run.json'
    if pointer.exists():
        return Path(json.loads(pointer.read_text())['run'])
    return ROOT / 'results/masac_v3_baseline_20261102'


def describe(root):
    cfg = json.loads((root / 'protocol.json').read_text())
    status_path = root / 'status.json'
    s = json.loads(status_path.read_text())
    completed, recovery_completed, error = (root / 'completion.json',
                                             root / 'recovery_completion.json',
                                             root / 'error.json')
    stamp = s.get('updated_at', status_path.stat().st_mtime)
    age = max(0., time.time() - stamp)
    clock = datetime.datetime.fromtimestamp(stamp, ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M:%S')
    if 'overall_total_new_steps' in s:
        overall = s['overall_completed_new_steps'] / s['overall_total_new_steps']
        label = '新增预算'
    else:
        arms, seeds = cfg['arms'], cfg['seeds']
        index = arms.index(s['mode']) * len(seeds) + seeds.index(s['seed'])
        overall = (index * s['total_steps'] + s['step']) / (len(arms) * len(seeds) * s['total_steps'])
        label = '总预算'
    finished = completed.exists() or recovery_completed.exists()
    phase = ('恢复分片已完成' if recovery_completed.exists() else '已完成') if finished else (
        '失败' if error.exists() else s.get('phase', 'training'))
    health = f'最后更新 {age:.0f}s 前；需检查进程' if age > 60 and not finished and not error.exists() else phase
    eta = s.get('eta_s')
    eta_text = f'{eta/60:.1f}min' if eta is not None else '计算中'
    val_step = s.get('last_evaluation_step')
    latest = root / 'latest_evaluation.json'
    if val_step is None and latest.exists():
        evaluation = json.loads(latest.read_text())
        if (evaluation.get('mode'), evaluation.get('seed')) == (s['mode'], s['seed']):
            val_step = evaluation['step']
    val = s.get('validation_success')
    validation = f'{val:.1%}' if val is not None else '尚未评估'
    stage = f' | stage={s["actual_stage"]}' if s.get('actual_stage') else ''
    if s.get('planned_stage') and s['planned_stage'] != s.get('actual_stage'):
        stage += f'（下回合切换至 {s["planned_stage"]}）'
    line = (f'{clock} CST | {s["mode"]} seed={s["seed"]} '
            f'{s["step"]:,}/{s["total_steps"]:,} | {label} {overall:.1%} '
            f'| updates={s["updates"]:,}{stage} | actor={s.get("actor_loss", 0):+.4f} '
            f'critic={s.get("critic_loss", 0):.4f} alpha={s.get("alpha", .1):.4f} '
            f'| {s["steps_per_s"]:.1f} steps/s seedETA={eta_text} '
            f'| val@{val_step if val_step is not None else "?"}={validation} | {health}')
    return line, finished or error.exists(), error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--run', type=Path, help='Result directory; defaults to latest active V3 run')
    parser.add_argument('--interval', type=float, default=5.)
    args = parser.parse_args()
    print('MASAC V3 · 只读进度；Ctrl-C 只关闭查看器\n', flush=True)
    try:
        while True:
            root = args.run.resolve() if args.run else active_run()
            try:
                line, finished, error = describe(root)
                print(line, flush=True)
                if error.exists():
                    print(error.read_text(), flush=True)
                if finished:
                    break
            except (FileNotFoundError, json.JSONDecodeError):
                print(f'等待完整进度记录：{root}', flush=True)
            if not args.watch:
                break
            time.sleep(max(.5, args.interval))
    except KeyboardInterrupt:
        print('\n查看器已关闭。训练状态未改变。')


if __name__ == '__main__':
    main()
