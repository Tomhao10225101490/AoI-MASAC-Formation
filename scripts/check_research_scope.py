import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'scope-lock.yaml').read_text())
assert s['RESEARCH_DIRECTION']=='FROZEN'
assert s['project']=='MASAC-3D-UAV-Path-Planning'
assert s['core']=='Learning-Based Multi-UAV 3D Path Planning with Multi-Agent Soft Actor-Critic'
assert s['candidate_status']=='unverified'
for name in ('AGENTS.md','RESEARCH_SCOPE.md','RESEARCH_DECISIONS.md','NOVELTY_ANALYSIS.md'):
    assert (root/name).is_file(),name
assert 'RESEARCH_DIRECTION = FROZEN' in (root/'RESEARCH_SCOPE.md').read_text()
print('Scope structure consistent; human scientific review remains required.')
