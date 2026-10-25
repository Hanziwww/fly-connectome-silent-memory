"""Frozen-model study records, bounded storage and exact readout algebra."""
from pathlib import Path
import contextlib
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
for p in (ROOT / 'flybrain', ROOT / 'sokoban', STUDY / 'code'):
    if str(p) not in sys.path:
        sys.path.append(str(p))
FIELDS = ('spikes', 'g', 'v', 'adapt', 'res')
PHASES = (('before', 'initial.pt'), ('mid', 'mid.pt'), ('after', 'model.pt'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_metadata(path):
    """Read the indented result schema while streaming past bulky neural arrays.

    This reader is only for registration/acceptance counts, never for feature analysis.
    Original JSONs remain unmodified. Array closing indentation must match exactly.
    """
    import re
    output=[];skip_indent=None
    pattern=re.compile(r'^(\s*)"(features|logits|decisions|probes|observations|records)": \[$')
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if skip_indent is not None:
                if line.rstrip() in (skip_indent+']',skip_indent+'],'):
                    output.append(skip_indent+']'+(',' if line.rstrip().endswith(',') else '')+'\n');skip_indent=None
                continue
            match=pattern.match(line.rstrip())
            output.append(line)
            if match:skip_indent=match.group(1)
    if skip_indent is not None:raise ValueError('Incomplete metadata array in '+str(path))
    return json.loads(''.join(output))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensors_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def scientific_sources():
    paths = [ROOT / 'sokoban/lib/brain_model.py', ROOT / 'sokoban/readout_models.py',
             STUDY / 'code/backend.py', STUDY / 'code/memory_tasks.py', ROOT / 'sokoban/lib/game.py']
    paths += sorted((ROOT / 'flybrain/flybrain').glob('*.py'))
    paths += [STUDY / 'remote-supplement-20261003' / n for n in ('supp_tasks.py', 'supp_experiments.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def code_manifest():
    return {p.name: digest(p) for p in sorted(CODE.glob('*.py')) if not p.name.startswith('test_')}


@contextlib.contextmanager
def file_lock(path, blocking=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            f.seek(0); f.write(b'0'); f.flush(); f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield f
    finally:
        f.close()


def check_stop():
    if (SESSION / 'STOP').exists():
        raise InterruptedError('Mechanism study STOP requested')
    if shutil.disk_usage(SESSION).free < 50 * 2**30:
        raise RuntimeError('resource_gate: less than 50 GiB free disk')


def save_npz(path, **arrays):
    """Reserve uncompressed size before writing; four workers share the cap."""
    check_stop(); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    if any(a.dtype.hasobject for a in arrays.values()):
        raise ValueError('Object arrays are not scientific storage')
    estimate = sum(a.nbytes for a in arrays.values()) + 8192
    relative = path.relative_to(SESSION).as_posix()
    ledger_path = SESSION / 'storage.json'
    with file_lock(SESSION / 'storage.lock'):
        ledger = read(ledger_path) if ledger_path.exists() else {'files': {}, 'reserved': {}}
        if relative in ledger['reserved']:
            raise RuntimeError('Existing interrupted file reservation requires diagnosis: ' + relative)
        total = sum(ledger['files'].values()) - ledger['files'].get(relative, 0)
        if total + sum(x['bytes'] for x in ledger['reserved'].values()) + estimate > 120 * 2**30:
            raise RuntimeError('resource_gate: 120 GiB derived-data budget')
        ledger['reserved'][relative] = {'bytes': estimate, 'pid': os.getpid()}
        write(ledger_path, ledger)
    tmp = path.with_name(path.name + '.partial')
    try:
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
        with file_lock(SESSION / 'storage.lock'):
            ledger = read(ledger_path); ledger['files'][relative] = path.stat().st_size
            ledger['reserved'].pop(relative); write(ledger_path, ledger)
    except BaseException:
        # Keep interrupted files and reservations for an explicit diagnosis.
        raise


def load_pair(path):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    pair = saved.get('pair', saved)
    if 'core' not in pair or 'head' not in pair:
        raise ValueError('Expected an existing brain and linear readout checkpoint')
    return pair


def head_arrays(pair):
    return {k: v.detach().cpu().numpy() for k, v in pair['head'].items()}


def standardized(features, head):
    return np.clip((np.asarray(features) - head['mean']) / head['scale'], -10, 10)


def logits(features, head):
    return standardized(features, head) @ head['proposal.weight'].T + head['proposal.bias']


def decision_basis(head):
    w = np.asarray(head['proposal.weight'], np.float64)
    centered = w - w.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-10, 1e-12)))
    if rank > 3:
        raise ArithmeticError('Four-action centered readout exceeds rank three')
    return vt[:rank].T, singular[:rank]


def contribution_decomposition(old_f, new_f, old_h, new_h):
    q00, q10 = logits(old_f, old_h), logits(new_f, old_h)
    q01, q11 = logits(old_f, new_h), logits(new_f, new_h)
    brain, head = q10 - q00, q01 - q00
    interaction = q11 - q10 - q01 + q00
    residual = np.max(np.abs((brain + head + interaction) - (q11 - q00)))
    return dict(q00=q00, q10=q10, q01=q01, q11=q11, brain=brain, head=head,
                interaction=interaction, reconstruction_error=float(residual))


def unique_phases(model):
    return [(p, SESSION / model['checkpoints'][p]['path']) for p, _ in PHASES if p in model['checkpoints']]


def model_by_id(model_id):
    return next(m for m in read(SESSION / 'models.json') if m['id'] == model_id)


def progress(out, **fields):
    write(Path(out) / 'progress.json', dict(updated_unix=time.time(), pid=os.getpid(), **fields))
