import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_masac_v3_curriculum import analyze, wait_for_run, write_new


def put(path, value):
    path.write_text(json.dumps(value))


def make_runs(tmp_path):
    run, baseline = tmp_path/'curriculum', tmp_path/'baseline'
    run.mkdir(); baseline.mkdir()
    maps = list(range(800001, 800007))
    for root in (run, baseline):
        put(root/'completion.json', {'complete':True})
        put(root/'protocol.json', {'steps':100000,'seeds':[411,422,433], 'arms':['standard'],
            'validation_map_seeds':maps, 'vmax':10., 'smoke':False})
        for seed in (411,422,433):
            d = root/f'standard_{seed}'; d.mkdir()
            rows = [dict(map_seed=m, layout='urban', success=False, static_collision=m==maps[0],
                boundary_collision=m==maps[0], dynamic_collision=False, uav_collision=False,
                remaining_distance=42., path_length=60., time=5.) for m in maps]
            put(d/'evaluation_100000.json', rows)
            for m in maps:
                put(d/f'eval_100000_{m}_map.json', {'seed':m, 'layout':'urban'})
            if root == run:
                put(d/'audit.json', {'total_steps':100000,'replay_checkpoint_alignment_step':100000,'replay_size':2})
                (d/'checkpoint_100000.pt').write_bytes(b'checkpoint')
                episode = dict(kind='episode',actual_stage='short_edge',episode_start_step=0,step=3,success=False,terminated=True,
                    truncated=False,static_collision=True,boundary_collision=True,dynamic_collision=False,
                    uav_collision=False,remaining_distance=40.,time=1.)
                (d/'training.jsonl').write_text(json.dumps(episode)+'\n')
                obs = np.zeros((2,3,162), dtype=np.float32)
                obs[:,:,0] = 5./240.
                obs[0,0,0] = 1./240.
                np.savez_compressed(d/'replay_final.npz', obs=obs)
    return run, baseline


def test_analysis_counts_overlapping_flags_and_replay_states(tmp_path):
    run, baseline = make_runs(tmp_path)
    result = analyze(run, baseline)
    assert len(result['seeds']) == 3
    first = result['seeds'][0]
    assert first['curriculum_final']['episodes'] == 6
    assert first['curriculum_final']['collision_flag_counts']['static_collision'] == 1
    assert first['curriculum_final']['collision_flag_counts']['boundary_collision'] == 1
    assert first['completed_training_episodes_by_stage']['short_edge']['completed_episodes'] == 1
    coverage = first['final_replay_coverage']
    assert coverage['agent_states'] == 6
    assert coverage['agent_states_below_distance_m']['4'] == 1
    assert coverage['agent_states_with_goal_capture'] == 1


def test_requires_completion_and_never_overwrites(tmp_path, monkeypatch):
    run, baseline = make_runs(tmp_path)
    (run/'completion.json').unlink()
    with pytest.raises(ValueError, match='no completed batch'):
        analyze(run, baseline)
    monkeypatch.setattr('scripts.analyze_masac_v3_curriculum.process_alive', lambda _: False)
    with pytest.raises(RuntimeError, match='without a completion marker'):
        wait_for_run(run, 123, .01, .1)
    out = tmp_path/'analysis.json'
    write_new(out, {'ok':True})
    with pytest.raises(FileExistsError):
        write_new(out, {'overwrite':True})
    assert json.loads(out.read_text()) == {'ok':True}
