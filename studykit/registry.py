"""Records of one registered study: protocol, registration, schedule, task lock, amendments, STOP, locks, events
and code provenance.

Generalises studies/brain-sokoban-v1/code/registry.py (frozen; never imported here) from one hard-coded study to
any folder ``studies/<name>``:

- ``Study.protocol(**design)`` builds the protocol record (the frozen study's ``protocol()``), with the SHA-256 of
  declared input sources; ``save_protocol`` writes it while it is a draft and ``register`` freezes it.
- ``Study.provenance()`` hashes the project source behind a result (the frozen study's ``provenance()``): the
  study's own code, this package and every project module currently loaded, snapshotted once under
  ``code-snapshots/<key>/``.

Records are written atomically; ``events.jsonl`` and ``amendments/`` are append-only; a registered protocol is
never rewritten (changes are amendments). The per-study worker lock ``studies/<name>/worker.lock`` replaces the
machine-wide ``runtime/worker.lock`` of the frozen study.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
FROZEN_STUDIES = frozenset({'brain-sokoban-v1'})
STATUSES = ('draft', 'registered', 'closed')
STUDY_NAME = re.compile(r'[a-z0-9][a-z0-9-]*[a-z0-9]')
JOB_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
LOCK_ENV = 'STUDYKIT_LOCK_HOLDER'
EXIT_STOPPED = 75  # a worker that stopped at a safe point because of STOP (EX_TEMPFAIL); the queue reruns it on resume
# Directories whose modules are third-party or interpreter code, never project source.
NOT_SOURCE = frozenset({'.venv', '.venv-linux', 'site-packages', 'dist-packages', '__pycache__', '.git'})


class StudyError(RuntimeError):
    """A study record is missing, malformed or inconsistent with its registration."""


class StudyLocked(StudyError):
    """Another process holds the study's worker lock (or a job lock)."""


class StudyStopped(InterruptedError):
    """The study's STOP file exists."""


# ---------------------------------------------------------------- records

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    """Write JSON atomically (temporary file, then replace), with LF line ends on every platform."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8',
                   newline='\n')
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader still has the target open
            if attempt == 19:
                raise
            time.sleep(0.05)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------- locks

_HELD = {}  # lock path -> [open handle, depth]; a process re-enters its own locks


def _lock(handle, blocking):
    if os.name == 'nt':
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))


def _unlock(handle):
    # POSIX: closing releases the flock once no other descriptor of it is open; an explicit LOCK_UN would also
    # release it for a child process that inherited the descriptor (a queue worker), which must keep it.
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def _lock_key(path):
    return os.path.normcase(str(Path(path).resolve()))


@contextlib.contextmanager
def file_lock(path, blocking=False):
    """Exclusive advisory lock on ``path`` (created if missing), re-entrant within a process.

    Raises StudyLocked when another process holds it and ``blocking`` is false. Yields the open handle; on POSIX a
    child process that inherits its descriptor keeps the lock held after this process exits.
    """
    key = _lock_key(path)
    if key in _HELD:
        _HELD[key][1] += 1
        try:
            yield _HELD[key][0]
        finally:
            _HELD[key][1] -= 1
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    try:
        if os.name == 'nt' and path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        _lock(handle, blocking)
    except OSError as error:
        handle.close()
        raise StudyLocked(f'{path} is held by another process') from error
    _HELD[key] = [handle, 1]
    try:
        yield handle
    finally:
        del _HELD[key]
        _unlock(handle)


def lock_state(path):
    """'absent', 'free', 'held' (another process) or 'held_here' (this process), without creating the file."""
    path = Path(path)
    if not path.exists():
        return 'absent'
    if _lock_key(path) in _HELD:
        return 'held_here'
    handle = path.open('rb')
    try:
        _lock(handle, blocking=False)
    except OSError:
        handle.close()
        return 'held'
    _unlock(handle)
    return 'free'


# ---------------------------------------------------------------- provenance helpers

def project_module_files(root=ROOT, modules=None):
    """Source files (.py) of loaded modules that live inside ``root``, outside virtual environments.

    ``modules``: module objects or file paths; default every module in sys.modules.
    """
    root = Path(root).resolve()
    items = list(sys.modules.values()) if modules is None else list(modules)
    found = set()
    for item in items:
        name = item if isinstance(item, (str, os.PathLike)) else getattr(item, '__file__', None)
        if not name:
            continue
        path = Path(name).resolve()
        if path.suffix != '.py' or not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if NOT_SOURCE.intersection(rel.parts) or rel.parts[:2] == ('MIGRATION', 'python'):
            continue
        found.add(path)
    return sorted(found)


def _zip_bytes(root, paths):
    """Deterministic zip of ``paths`` (fixed timestamps, sorted order)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(paths, key=lambda q: q.relative_to(root).as_posix()):
            info = zipfile.ZipInfo(p.relative_to(root).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())
    return buffer.getvalue()


# ---------------------------------------------------------------- study

class Study:
    """One study folder, ``<root>/studies/<name>``."""

    def __init__(self, path, root=None):
        self.path = Path(path).resolve()
        self.name = self.path.name
        if not STUDY_NAME.fullmatch(self.name):
            raise StudyError(f'invalid study name {self.name!r} (lower-case letters, digits and hyphens)')
        if root is not None:
            self.root = Path(root).resolve()
        else:
            self.root = self.path.parents[1] if self.path.parent.name == 'studies' else ROOT
        try:
            self.path.relative_to(self.root)
        except ValueError:
            raise StudyError(f'{self.path} is not inside {self.root}') from None

    @classmethod
    def named(cls, name, root=None):
        if not isinstance(name, str) or not STUDY_NAME.fullmatch(name):
            raise StudyError(f'invalid study name {name!r} (lower-case letters, digits and hyphens)')
        root = Path(root).resolve() if root is not None else ROOT
        return cls(root / 'studies' / name, root)

    def __repr__(self):
        return f'Study({self.relative(self.path)!r})'

    @property
    def frozen(self):
        return self.name in FROZEN_STUDIES

    def relative(self, path):
        return Path(path).resolve().relative_to(self.root).as_posix()

    # paths
    protocol_path = property(lambda self: self.path / 'protocol.json')
    registration_path = property(lambda self: self.path / 'registration.json')
    schedule_path = property(lambda self: self.path / 'schedule.json')
    tasks_dir = property(lambda self: self.path / 'tasks')
    task_lock_path = property(lambda self: self.path / 'tasks' / 'lock.json')
    amendments_dir = property(lambda self: self.path / 'amendments')
    code_dir = property(lambda self: self.path / 'code')
    snapshots_dir = property(lambda self: self.path / 'code-snapshots')
    runs_dir = property(lambda self: self.path / 'runs')
    events_path = property(lambda self: self.path / 'events.jsonl')
    supervisor_path = property(lambda self: self.path / 'supervisor.json')
    stop_path = property(lambda self: self.path / 'STOP')
    lock_path = property(lambda self: self.path / 'worker.lock')

    def run_dir(self, job_id):
        if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
            raise StudyError(f'invalid job id {job_id!r}')
        return self.runs_dir / job_id

    # ------------------------------------------------------------ protocol

    def protocol(self, sources=(), status='draft', **design):
        """Protocol record: design fields plus the SHA-256 of each declared input source (project-relative paths)."""
        if status not in STATUSES:
            raise StudyError(f'unknown protocol status {status!r}')
        hashes = {}
        for rel in sources:
            path = (self.root / rel).resolve()
            path.relative_to(self.root)
            hashes[path.relative_to(self.root).as_posix()] = digest(path)
        return dict(schema_version=SCHEMA_VERSION, study=self.name, **design, sources=hashes, status=status,
                    created_unix=time.time())

    def load_protocol(self):
        if not self.protocol_path.exists():
            raise StudyError(f'{self.relative(self.protocol_path)} is missing')
        value = read(self.protocol_path)
        if not isinstance(value, dict):
            raise StudyError('protocol.json must hold an object')
        if value.get('study') != self.name:
            raise StudyError(f"protocol.json names study {value.get('study')!r}, not {self.name!r}")
        if value.get('schema_version') != SCHEMA_VERSION:
            raise StudyError(f"protocol.json schema_version {value.get('schema_version')!r} is not {SCHEMA_VERSION}")
        if value.get('status') not in STATUSES:
            raise StudyError(f"protocol.json status {value.get('status')!r} is not one of {STATUSES}")
        return value

    def save_protocol(self, value):
        """Write a draft protocol. A registered protocol is never rewritten; changes go through ``amend``."""
        if self.protocol_path.exists() and read(self.protocol_path).get('status') != 'draft':
            raise StudyError('the protocol is registered; record changes with Study.amend')
        if value.get('study') != self.name or value.get('status') != 'draft':
            raise StudyError("a saved protocol must name this study and have status 'draft'")
        write(self.protocol_path, value)

    # ------------------------------------------------------------ schedule and task lock

    def schedule(self):
        """Jobs of schedule.json, checked: unique safe ids, known dependencies, no cycles."""
        if not self.schedule_path.exists():
            raise StudyError(f'{self.relative(self.schedule_path)} is missing')
        jobs = read(self.schedule_path)
        if not isinstance(jobs, list):
            raise StudyError('schedule.json must hold a list of jobs')
        check_jobs(jobs)
        return jobs

    def lock_tasks(self, **meta):
        """Write tasks/lock.json with the SHA-256 of every other tasks/*.json file (the frozen study's lock format)."""
        files = {p.name: digest(p) for p in sorted(self.tasks_dir.glob('*.json')) if p.name != 'lock.json'}
        value = dict(schema_version=SCHEMA_VERSION, study=self.name, created_unix=time.time(), files=files, **meta)
        write(self.task_lock_path, value)
        return value

    def check_task_lock(self):
        """Compare tasks/*.json with tasks/lock.json: missing, changed and unlocked (present but not listed) files."""
        if not self.task_lock_path.exists():
            return dict(ok=False, error='tasks/lock.json is missing', missing=[], changed=[], unlocked=[])
        locked = read(self.task_lock_path).get('files', {})
        present = {p.name: p for p in self.tasks_dir.glob('*.json') if p.name != 'lock.json'}
        missing = sorted(n for n in locked if n not in present)
        changed = sorted(n for n in locked if n in present and digest(present[n]) != locked[n])
        unlocked = sorted(n for n in present if n not in locked)
        return dict(ok=not (missing or changed or unlocked), files=len(locked), missing=missing, changed=changed,
                    unlocked=unlocked)

    # ------------------------------------------------------------ amendments (append-only, hash-chained)

    def amendments(self):
        """Amendment records in order; raises StudyError if numbering or the hash chain is broken."""
        files = sorted(self.amendments_dir.glob('[0-9][0-9][0-9][0-9]-*.json'))
        records, previous = [], None
        for number, path in enumerate(files, start=1):
            value = read(path)
            if value.get('number') != number or not path.name.startswith(f'{number:04d}-'):
                raise StudyError(f'amendment numbering is broken at {path.name}')
            if value.get('previous_sha256') != previous:
                raise StudyError(f'amendment hash chain is broken at {path.name} (an earlier amendment changed)')
            previous = digest(path)
            records.append(dict(value, file=path.name, sha256=previous))
        return records

    def amend(self, reason, changes, **extra):
        """Append amendment NNNN; it records the SHA-256 of the previous one, so later edits are detectable."""
        with file_lock(self.amendments_dir / '.lock', blocking=True):
            existing = self.amendments()
            number = len(existing) + 1
            slug = re.sub(r'[^a-z0-9]+', '-', reason.lower()).strip('-')[:40] or 'amendment'
            path = self.amendments_dir / f'{number:04d}-{slug}.json'
            value = dict(schema_version=SCHEMA_VERSION, study=self.name, number=number, reason=reason,
                         changes=changes, previous_sha256=existing[-1]['sha256'] if existing else None,
                         protocol_sha256=digest(self.protocol_path) if self.protocol_path.exists() else None,
                         created_unix=time.time(), **extra)
            data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, 'O_BINARY', 0))
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
        self.event('amended', number=number, file=path.name)
        return path

    # ------------------------------------------------------------ registration

    def integrity(self):
        """Current SHA-256 of the registered records (None where a file is missing)."""
        def h(path):
            return digest(path) if path.exists() else None
        chain = self.amendments()
        return dict(protocol_sha256=h(self.protocol_path), schedule_sha256=h(self.schedule_path),
                    task_lock_sha256=h(self.task_lock_path), amendments=len(chain),
                    amendments_head_sha256=chain[-1]['sha256'] if chain else None)

    def register(self, **extra):
        """Freeze a draft: protocol status 'registered' and registration.json with the record hashes."""
        protocol = self.load_protocol()
        if protocol['status'] != 'draft':
            raise StudyError(f"the protocol is already {protocol['status']}")
        self.schedule()
        lock = self.check_task_lock()
        if not lock['ok']:
            raise StudyError(f'tasks/lock.json does not match tasks/: {lock}')
        protocol.update(status='registered', registered_unix=time.time())
        write(self.protocol_path, protocol)
        record = dict(schema_version=SCHEMA_VERSION, study=self.name, registered_unix=protocol['registered_unix'],
                      **self.integrity(), **extra)
        write(self.registration_path, record)
        self.event('registered', protocol_sha256=record['protocol_sha256'], schedule_sha256=record['schedule_sha256'])
        return record

    def check_registration(self):
        """Compare protocol, schedule and task lock with registration.json. Amendments may only be appended."""
        if not self.registration_path.exists():
            return dict(registered=False, ok=False, changed=[], error='not registered (no registration.json)')
        record = read(self.registration_path)
        current = self.integrity()
        changed = [k for k in ('protocol_sha256', 'schedule_sha256', 'task_lock_sha256') if record.get(k) != current[k]]
        if current['amendments'] < record.get('amendments', 0):
            changed.append('amendments')
        return dict(registered=True, ok=not changed, changed=changed, amendments_since=current['amendments'] -
                    record.get('amendments', 0))

    # ------------------------------------------------------------ control: STOP, locks, events

    def stopped(self):
        return self.stop_path.exists()

    def check_stop(self):
        if self.stopped():
            raise StudyStopped(f'{self.relative(self.stop_path)} exists')

    @contextlib.contextmanager
    def worker_lock(self, blocking=False):
        """Per-study worker lock. While it is held, LOCK_ENV tells child processes that their parent holds it."""
        with file_lock(self.lock_path, blocking) as handle:
            previous = os.environ.get(LOCK_ENV)
            os.environ[LOCK_ENV] = f'{self.name}:{os.getpid()}'
            try:
                yield handle
            finally:
                if previous is None:
                    os.environ.pop(LOCK_ENV, None)
                else:
                    os.environ[LOCK_ENV] = previous

    def held_by_parent(self):
        """True when this process was started by the holder of this study's worker lock (e.g. a queue worker)."""
        return os.environ.get(LOCK_ENV) == f'{self.name}:{os.getppid()}'

    def lock_state(self):
        return lock_state(self.lock_path)

    def event(self, name, **fields):
        append(self.events_path, dict(event=name, study=self.name, time=time.time(), **fields))

    # ------------------------------------------------------------ provenance

    def provenance(self, modules=None, extra=()):
        """Hash the source behind a result and snapshot it once under code-snapshots/<key>/ (source.zip, manifest.json).

        Covers every .py in the study's code/ folder and in studykit/, every project module loaded in this process
        (``modules`` overrides sys.modules; see project_module_files) and ``extra`` files. Call it once the process has
        imported what it uses. Processes that load different modules get different keys, so check a recorded key with
        ``changed_files(key)`` instead of recomputing it elsewhere.
        """
        paths = set(self.code_dir.glob('*.py')) | set((self.root / 'studykit').glob('*.py'))
        paths |= set(project_module_files(self.root, modules))
        paths |= {Path(p).resolve() for p in extra}
        paths = {p.resolve() for p in paths}
        manifest = {p.relative_to(self.root).as_posix(): digest(p) for p in sorted(paths)}
        key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        folder = self.snapshots_dir / key
        if not (folder / 'manifest.json').exists():
            tmp = self.snapshots_dir / f'.{key}.{os.getpid()}.partial'
            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / 'source.zip').write_bytes(_zip_bytes(self.root, paths))
            write(tmp / 'manifest.json', manifest)
            try:
                os.rename(tmp, folder)
            except OSError:
                if not (folder / 'manifest.json').exists():
                    raise
                shutil.rmtree(tmp, ignore_errors=True)  # another process stored the same snapshot first
        return key

    def snapshot(self, key):
        path = self.snapshots_dir / key / 'manifest.json'
        if not path.exists():
            raise StudyError(f'code snapshot {key} is missing')
        return read(path)

    def changed_files(self, key):
        """Files of snapshot ``key`` that are now missing or different ([] means the recorded code is unchanged)."""
        changed = []
        for rel, sha in sorted(self.snapshot(key).items()):
            path = self.root / rel
            if not path.is_file():
                changed.append(dict(path=rel, reason='missing'))
            elif digest(path) != sha:
                changed.append(dict(path=rel, reason='changed'))
        return changed


def check_jobs(jobs):
    """Validate a job list: unique safe ids, dependencies that exist, no dependency cycle."""
    ids = []
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get('id'), str) or not JOB_ID.fullmatch(job['id']):
            raise StudyError(f'invalid job entry {job!r}')
        ids.append(job['id'])
    if len(set(ids)) != len(ids):
        raise StudyError('duplicate job ids in the schedule')
    deps = {j['id']: list(j.get('dependencies', [])) for j in jobs}
    unknown = sorted({d for ds in deps.values() for d in ds} - set(ids))
    if unknown:
        raise StudyError(f'unknown dependencies {unknown}')
    remaining, done = dict(deps), set()
    while remaining:
        ready = [k for k, ds in remaining.items() if set(ds) <= done]
        if not ready:
            raise StudyError(f'dependency cycle among {sorted(remaining)}')
        for k in ready:
            done.add(k)
            del remaining[k]
    return jobs
