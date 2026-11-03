import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import train_masac_v4_l0 as runner
from uav_planning.v4_environment import V4Mission


def config():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root/'configs/masac_v4/l0_protocol.json').read_text())


@pytest.mark.parametrize('failure', ['no_witness', 'hash_mismatch'])
def test_training_map_gate_records_then_rejects(failure, tmp_path, monkeypatch):
    def bad_reference(scene_config, seed):
        env=V4Mission(scene_config)
        env.reset(seed)
        return dict(scene={'scene_sha256':('different' if failure=='hash_mismatch' else env.scene.scene_hash)},
                    joint_reference_success=failure!='no_witness', attempts=[])
    monkeypatch.setattr(runner,'joint_reference',bad_reference)
    with pytest.raises(ValueError,match='reference failed or scene hash mismatched'):
        runner.audited_episode('urban_canyon',config(),7110000,0,tmp_path)
    rows=(tmp_path/'training_map_audit.jsonl').read_text().splitlines()
    assert len(rows)==1
    row=json.loads(rows[0])
    assert row['map_seed']==7110000 and row['episode']==0
    assert row['reference_scene_hash_matches'] == (failure!='hash_mismatch')
    assert row['joint_reference_success'] == (failure!='no_witness')


def test_training_map_gate_preserves_global_rng_and_returns_only_actor_observation(tmp_path):
    np.random.seed(123); random.seed(123); torch.manual_seed(123)
    np_before=np.random.get_state(); py_before=random.getstate(); torch_before=torch.get_rng_state().clone()
    env,obs=runner.audited_episode('urban_canyon',config(),7110000,0,tmp_path)
    np_after=np.random.get_state()
    assert np_before[0] == np_after[0]
    np.testing.assert_array_equal(np_before[1],np_after[1])
    assert np_before[2:] == np_after[2:]
    assert py_before == random.getstate()
    assert torch.equal(torch_before,torch.get_rng_state())
    assert obs.shape==(3,162)
    row=json.loads((tmp_path/'training_map_audit.jsonl').read_text())
    assert row['reference_scene_hash_matches'] and row['joint_reference_success']
    assert row['scene_sha256']==env.scene.scene_hash
    assert 'paths' not in row and 'witness' not in row
