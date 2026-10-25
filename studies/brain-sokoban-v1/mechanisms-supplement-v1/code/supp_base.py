"""Independent frozen follow-up; original experiments remain read-only."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import time
import numpy as np

CODE = Path(__file__).resolve().parent
SESSION = CODE.parent
STUDY = SESSION.parent
ROOT = STUDY.parents[1]
SOURCE = STUDY / 'mechanisms-v1'
sys.path.insert(0, str(SOURCE / 'code'))
from mechanism_base import read, write, digest, object_hash, file_lock, load_pair

MODEL_IDS = ['remote-dynamic-M0-head_only', 'remote-dynamic-M0-joint',
             'supp-M0-visibility', 'supp-S1-ring-brain', 'supp-S2-context-brain']
TIMINGS = [(d, p) for d in (40, 160, 640) for p in (0, 4)]
PARTS = {'voltage': ('v',), 'current': ('g', 'g_slow', 'g_slow_cls'),
         'adaptation': ('adapt',), 'continuous': ('v', 'g', 'g_slow', 'g_slow_cls', 'adapt')}


def models():
    return read(SESSION / 'models.json')


def get_model(name):
    return next(m for m in models() if m['id'] == name)


def task_for(model):
    return next(t for t in read(SOURCE / 'inputs/memory-tasks.json') if t['id'] == model['task'])


def checkpoint(model):
    return SOURCE / model['checkpoints']['after']['path']


def code_hashes():
    return {p.name: digest(p) for p in sorted(CODE.glob('*.py')) if not p.name.startswith('test_')}


def check_stop():
    if (SESSION / 'STOP').exists():
        raise InterruptedError('Frozen supplement STOP requested')
    if shutil.disk_usage(SESSION).free < 50 * 2**30:
        raise RuntimeError('resource_gate: 50 GiB free disk reserve')


def progress(out, **fields):
    write(Path(out) / 'progress.json', dict(updated_unix=time.time(), pid=os.getpid(), **fields))


def save_npz(path, **arrays):
    check_stop()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    if any(v.dtype.hasobject for v in arrays.values()):
        raise TypeError('Object arrays are forbidden')
    estimate = sum(v.nbytes for v in arrays.values()) + 8192
    key = path.relative_to(SESSION).as_posix()
    with file_lock(SESSION / 'storage.lock'):
        ledger = read(SESSION / 'storage.json') if (SESSION / 'storage.json').exists() else {'files': {}, 'reserved': {}}
        if key in ledger['reserved']:
            raise RuntimeError('Interrupted write requires diagnosis: ' + key)
        if sum(ledger['files'].values()) + sum(ledger['reserved'].values()) + estimate > 20 * 2**30:
            raise RuntimeError('resource_gate: 20 GiB supplement data budget')
        ledger['reserved'][key] = estimate; write(SESSION / 'storage.json', ledger)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)
    with file_lock(SESSION / 'storage.lock'):
        ledger = read(SESSION / 'storage.json'); ledger['reserved'].pop(key)
        ledger['files'][key] = path.stat().st_size; write(SESSION / 'storage.json', ledger)


def make_manifest(out):
    files = {p.relative_to(out).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
             for p in Path(out).rglob('*') if p.is_file() and p.name not in
             ('manifest.json', 'worker.log', 'process.json') and not p.name.endswith(('.partial', '.tmp'))}
    write(Path(out) / 'manifest.json', dict(files=files, total_bytes=sum(v['bytes'] for v in files.values())))


def prepare():
    if (SESSION / 'plan.json').exists():
        plan = read(SESSION / 'plan.json')
        assert plan['code_hashes'] == code_hashes(), 'Registered code changed; preserve and diagnose'
        return plan
    selected = [m for m in read(SOURCE / 'models.json') if m['id'] in MODEL_IDS]
    assert len(selected) == 5
    write(SESSION / 'models.json', selected)
    groups = read(SOURCE / 'runs/discover-memory/groups.json')
    write(SESSION / 'groups.json', groups)
    baseline_files = [SOURCE / 'inputs/memory-tasks.json', SOURCE / 'neurons/index.npz']
    for m in selected:
        baseline_files += [checkpoint(m), SOURCE / 'runs' / ('record-' + m['id']) / 'result.json']
    import mechanism_base
    sources = {**mechanism_base.scientific_sources(), **{
        p.relative_to(ROOT).as_posix(): digest(p) for p in (SOURCE / 'code').glob('*.py')}}
    jobs = [dict(id='preflight-' + m, kind='preflight', model=m, dependencies=[]) for m in MODEL_IDS]
    prerequisites = [j['id'] for j in jobs]
    for kind in ('selective', 'routes', 'timing'):
        jobs += [dict(id=kind + '-' + m, kind=kind, model=m, dependencies=prerequisites) for m in MODEL_IDS]
    plan = dict(schema=1, registered_unix=time.time(), mode='frozen_replay_and_temporary_interventions',
        seed=10, optimizer_allowed=False, models=MODEL_IDS, jobs=jobs, gpu_lanes=[0, 1, 2, 3],
        gpu_allocator_gib=11, gpu_free_gate_gib=13, local_free_reserve_gib=50, data_budget_gib=20,
        original_training_stop_required=True, action_limit=512, timings=TIMINGS,
        selective=dict(parts=PARTS, windows=['write', 'hold', 'read'], donor_fractions=[.25, .5, 1.],
            controls=['norm_matched_scrambled_delta', 'same_cue', 'erase_restore'],
            continuous_only=True, discrete_buffers_and_clock_preserved=True),
        routes=dict(groups=[g['id'] for g in groups['groups'] if g['observed_count'] > 0],
            sides=['indices', 'controls'], strengths=[.25, .5, 1.],
            windows=['first_decision', 'all_decisions'], channels=['network', 'readout', 'both'],
            readout_replacement='per-feature checkpoint training mean; fixed pooling divisor 16'),
        timing=dict(total_postcue_ms=[80, 120, 160, 240, 320, 480, 640], prefix_windows=[2, 4, 8],
            inputs=['blank', 'neutral_stationary', 'neutral_walk_0', 'neutral_walk_1'],
            discovery_times=[80, 160, 320], held_times=[120, 240, 480, 640],
            held_input='neutral_walk_1', distractor='canonical cue geometry; all goals hidden; cue/context independent'),
        inference='paired histories; map families and one training seed; prior confirmation layouts are already known',
        source_hashes=sources, input_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in baseline_files},
        code_hashes=code_hashes(), protocol_sha256=digest(SESSION / 'protocol.en.md'))
    write(SESSION / 'plan.json', plan)
    return plan
