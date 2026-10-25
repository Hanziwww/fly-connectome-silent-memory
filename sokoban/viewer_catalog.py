"""Verified checkpoint registry shared by the viewer and intervention audit."""
from workspace import ARCHIVE, RUNS
import json
from bootstrap import HERE
from common import file_hash

OUT = ARCHIVE / 'campaign/run'
REPAIR = ARCHIVE / 'repair/20261001-b00-screen01'
INTERVENTION = ARCHIVE / 'repair/20261001-b00-intervention01'
PILOT_MODELS = dict(brain='After four levels', initial='Before learning', zero='Original parameters',
    stage1='After level 1', stage2='After level 2', stage3='After level 3', stage4='After level 4')
REPAIR_MODELS = {f'repair{seed}': f'Repair D · seed {seed}' for seed in (10, 11, 12)}
REPAIR_MODELS.update(repair_source='Before repair', zero='Original parameters')


def repair_bundle():
    verification = json.loads((REPAIR / 'verification.json').read_text())
    assert verification['status'] == 'verified'
    config = json.loads((REPAIR / 'config.json').read_text())
    paths = {'repair_source': REPAIR / 'source.pt'}
    assert file_hash(paths['repair_source']) == config['source_checkpoint_sha256']
    trials = {}
    for seed in (10, 11, 12):
        kind = f'repair{seed}'
        directory = REPAIR / f'trials/D-s{seed}'
        result = json.loads((directory / 'result.json').read_text())
        assert result['status'] == 'learned' and result['after']['solved'] == 1
        paths[kind] = directory / 'model.pt'
        assert file_hash(paths[kind]) == result['model_sha256']
        trials[kind] = result
    level = next(l for l in json.loads((REPAIR / 'levels.json').read_text()) if l['id'] == 'B00')
    return paths, trials, level
