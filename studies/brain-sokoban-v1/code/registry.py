"""Locked design, atomic records and source provenance for the research study."""
import hashlib
import json
import os
from pathlib import Path
import time

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
SOURCE = ROOT / 'runs/linear-joint/20261002-p05-01/model.pt'
TIERS = [(6, 1, 9, 12), (6, 1, 13, 16), (8, 1, 17, 21), (8, 2, 22, 27),
         (8, 2, 28, 33), (10, 2, 34, 39), (10, 3, 40, 47), (10, 3, 48, 60)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    paths = list((STUDY / 'code').glob('*.py')) + list((ROOT / 'sokoban').glob('*.py'))
    paths += list((ROOT / 'sokoban/lib').glob('*.py')) + list((ROOT / 'flybrain/flybrain').glob('*.py'))
    paths += [ROOT / 'workflow.py']
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}
    key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    folder = STUDY / 'code-snapshots' / key
    if not folder.exists():
        import zipfile
        folder.mkdir(parents=True)
        with zipfile.ZipFile(folder / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                z.write(p, p.relative_to(ROOT).as_posix())
        write(folder / 'manifest.json', manifest)
    return key


def protocol():
    return dict(schema_version=1, study='brain-sokoban-v1', seed=10, source=str(SOURCE.relative_to(ROOT)),
        source_sha256=digest(SOURCE), training_seeds=[10], tiers=TIERS,
        initialization='independent P05 forks; sequential branches explicitly labelled',
        success='autonomous solution within 512 actions; any length accepted',
        segments_seconds=[1200, 2400, 4800], minimum_margin_progress=.02,
        core_lr=.0001, head_lr=.0003, steps_per_assessment=4, allocator_gib=11, minimum_free_gib=13,
        delay_ms=[0, 20, 40, 80, 160, 320, 640], trained_delay_ms=[0, 40, 160],
        heldout_delay_ms=[20, 80, 320, 640], prefix_windows=[0, 1, 2, 4, 8],
        short_probe_prefix_windows=0,cross_step_prefix_windows=[1,2,4,8],
        trained_prefix_windows=[0, 1, 4], heldout_prefix_windows=[2, 8],
        memory_condition_selection='train conditions only; heldout delays/prefixes are evaluation only',
        representative_tasks=['D1-T0','D3-T0','D5-T0','D7-T0'],
        discovery_families='T0 in every tier', confirmation_families='T1 in every tier',
        activity_phases=['before','mid','after'], activity_precision='float32 measurements, recorded accumulation dtype',
        full_trace_state_indices=[0, .5, -1],
        inference_input='brain activities only except explicitly named board MLP / GRU baselines',
        independent_unit='map / map family; no inference to other training seeds',
        acceleration_acceptance='both calibration tasks learn and independently reload to solve; routes may differ',
        status='registered', created_unix=time.time())


def condition(name='joint'):
    value = dict(name=name, train_core=True, train_head=True, loss='recovery', schedule='suffix',
                 initialization='p05', structure='real', state_mode='reset', model='brain_linear')
    if name == 'frozen': value.update(train_core=False, train_head=False)
    elif name == 'brain_only': value['train_head'] = False
    elif name == 'head_only': value['train_core'] = False
    elif name == 'alternating': value['update_mode'] = 'alternating'
    elif name == 'no_recovery': value['loss'] = 'entry_only'
    elif name == 'entry_ce': value['loss'] = 'entry_ce'
    elif name == 'demonstration': value.update(schedule='demonstration', loss='demonstration_ce')
    elif name in ('neutral', 'rewired', 'weight_shuffle'):
        value.update(initialization='neutral', structure={'neutral':'real'}.get(name, name))
    elif name == 'board_mlp': value.update(initialization='neutral', model='board_mlp', train_core=False)
    elif name != 'joint': raise ValueError(name)
    return value


def make_schedule(levels):
    import random
    jobs = []
    def add(run_id, task, cond='joint', module='difficulty', parent=None, **extra):
        jobs.append(dict(id=run_id, task=task, condition=condition(cond), module=module,
                         parent=parent, seed=10, **extra))
    # Block on task, randomize conditions within a block. Keep the difficulty jobs
    # first to obtain all predeclared representative checkpoints for later analysis.
    for task in [l for l in levels if l['split']=='train']:
        add('difficulty-'+task['id'], task['id'])
    for tier in (1,3,5,7):
        arms = ['frozen','brain_only','head_only','alternating','no_recovery','entry_ce',
                'demonstration','neutral','rewired','weight_shuffle','board_mlp']
        random.Random(10+tier).shuffle(arms)
        for arm in arms: add(f'control-D{tier}-{arm}', f'D{tier}-T0', arm, 'controls')
    for tier in range(8):
        # First curriculum task is identical to a difficulty fork, reused exactly.
        if tier:
            add(f'curriculum-D{tier}',f'D{tier}-T0',module='curriculum',
                parent='difficulty-D0-T0' if tier==1 else f'curriculum-D{tier-1}')
    for tier in (1,3,5,7):
        add(f'transfer-D{tier}',f'D{tier}-H',module='transfer',parent=f'difficulty-D{tier}-T0')
        add(f'transfer-D{tier}-neutral',f'D{tier}-H','neutral',module='transfer')
    # Similar A/B = same map, independently chosen start. Different B = different family.
    pairs=[('similar','D1-T0','D1-T0-S0'),('different-medium','D3-T0','D3-T1'),
           ('different-hard','D5-T0','D7-T1')]
    for pair,a,b in pairs:
        add(f'memory-{pair}-B',b,module='parameter_memory',parent='difficulty-'+a, probe_tasks=[a])
        add(f'memory-{pair}-A2',a,module='parameter_memory',parent=f'memory-{pair}-B',
            restore_from='difficulty-'+a, probe_tasks=[a,b])
    return jobs
