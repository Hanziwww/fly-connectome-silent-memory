"""Input manifest: every input file read, hashed once per NTFS file identity (hard links share one entry). Files this run
generates (under the writable roots) are not inputs and are not recorded."""
import hashlib
import json
import os
from pathlib import Path

from . import paths

_entries = {}
_hashes = {}
_phase = 'unset'


def set_phase(name):
    global _phase
    _phase = name


def generated(path):
    """True for files under the two writable roots: this run's deliverables and intermediates, which the output manifest
    hashes. They are not inputs; recording one would also record a file identity that changes with every clean rebuild."""
    target = Path(path).resolve()
    return any(target == root or root in target.parents for root in (Path(r).resolve() for r in paths.WRITABLE))


def register(path, data):
    if generated(path):
        return
    rp = paths.rel(path)
    if rp in _entries:
        return
    st = os.stat(path)
    file_id = f'{st.st_dev:x}:{st.st_ino:x}'
    digest = _hashes.get(file_id)
    if digest is None:
        digest = _hashes[file_id] = hashlib.sha256(data).hexdigest()
    _entries[rp] = dict(path=rp, file_id=file_id, bytes=st.st_size, sha256=digest,
                        n_links=st.st_nlink, first_phase=_phase)


def save_phase(phase):
    target = paths.work(f'inputs/{phase}.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [e for e in _entries.values() if e['first_phase'] == phase]
    target.write_text(json.dumps(sorted(rows, key=lambda e: e['path']), indent=0), encoding='utf-8')


def merged():
    rows = {}
    folder = paths.work('inputs')
    if folder.exists():
        for f in sorted(folder.glob('*.json')):
            for e in json.loads(f.read_text(encoding='utf-8')):
                rows.setdefault(e['path'], e)
    for e in _entries.values():
        rows.setdefault(e['path'], e)
    return [rows[k] for k in sorted(rows)]
