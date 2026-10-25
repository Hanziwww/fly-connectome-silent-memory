"""Remote half of every transport operation (standard library only; Python 3.8 or newer).

ssh.py sends this file with each command and runs it with the remote's own interpreter, so it works before the
project environment exists and never depends on the state of the remote checkout:

    <agent_python> -B -X utf8 -c 'import base64,sys,zlib;exec(zlib.decompress(base64.b64decode(sys.argv[1])))' \
        <this file, zlib + base64> <parameters, base64 JSON with 'op' and 'root'>

Large inputs (file lists) arrive as JSON on stdin, a push bundle arrives as tar on stdin, and a pull leaves as tar on
stdout after STREAM_MARK. Every other operation prints one line, RESULT_MARK followed by JSON; a refusal exits with
REFUSED and {"refused": reason} on stderr. Local code imports this module for the shared path rules, so both sides
apply the same checks.

Operations: ping, manifest, receive, apply, send, status, stop, launch.
"""
import base64
import datetime
import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

RESULT_MARK = 'FLYBRAIN-AGENT-RESULT '
STREAM_MARK = b'FLYBRAIN-AGENT-STREAM\n'
MANIFEST_NAME = '.sync-manifest.json'
REFUSED = 3
BLOCK = 1 << 20
STUDY_NAME = re.compile(r'[a-z0-9][a-z0-9-]*[a-z0-9]')
PUSH_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*')
SAFE_ARG = re.compile(r'[A-Za-z0-9_.,:=+-]+')
FROZEN = ('brain-sokoban-v1',)
# Never transferred in either direction: credentials, machine state, version control and environments.
DENIED_TOP = ('local', 'runtime', '.git', '.sync', '.venv', '.venv-linux', 'remote.toml', 'ssh.txt')
DENIED_PARTS = ('.ssh', '.git', '__pycache__')
DENIED_NAMES = ('*known-hosts*', '*known_hosts*', 'id_rsa*', 'id_dsa*', 'id_ecdsa*', 'id_ed25519*', '*.pem', '*.ppk',
                '*.key', 'authorized_keys', 'ssh.txt', 'remote.toml')
TRANSIENT = ('.tmp', '.lock', '.partial', '.sync-partial', '.download')
# Inside studies/<name>/ (not frozen): written only by the remote queue, so pushes never overwrite them.
REMOTE_OWNED_DIRS = ('runs', 'code-snapshots')
REMOTE_OWNED_FILES = ('supervisor.json', 'events.jsonl', 'queue.log', 'STOP', 'worker.lock')
# Mirrored state that changes while a study runs; a pull may refresh these. All other files are evidence.
MUTABLE_NAMES = ('supervisor.json', 'events.jsonl', 'progress.json')
QUEUE_ACTIVE = ('running', 'stopping', 'waiting_resource')


class Refusal(Exception):
    """The operation is refused; nothing unsafe was done."""


# ---------------------------------------------------------------- shared path rules

def clean_relative(rel):
    """Normalised project-relative POSIX path; refuses absolute paths, '..', '.', empty parts and backslashes."""
    if not isinstance(rel, str) or not rel or '\\' in rel or '\x00' in rel:
        raise Refusal('invalid relative path %r' % (rel,))
    parts = rel.split('/')
    if rel.startswith('/') or any(p in ('', '.', '..') for p in parts):
        raise Refusal('path must be relative, without ".", ".." or empty parts: %r' % (rel,))
    return str(PurePosixPath(rel))


def denied(rel):
    """Why ``rel`` is never transferred, or None."""
    parts = rel.split('/')
    if parts[0] in DENIED_TOP:
        return 'top-level %s/ is never transferred' % parts[0]
    hit = [p for p in parts if p in DENIED_PARTS]
    if hit:
        return '%s is never transferred' % hit[0]
    name = parts[-1].lower()
    for pattern in DENIED_NAMES:
        if fnmatch.fnmatchcase(name, pattern):
            return 'credential or host-key file (%s)' % pattern
    return None


def study_part(rel):
    """(study, path inside the study folder) for paths under studies/<name>/, else None."""
    parts = rel.split('/')
    if len(parts) >= 3 and parts[0] == 'studies':
        return parts[1], '/'.join(parts[2:])
    return None


def remote_owned(rel, frozen=FROZEN):
    """True for outputs that only the remote queue writes (runs/, code-snapshots/, supervisor.json, ...)."""
    part = study_part(rel)
    if part is None or part[0] in frozen:
        return False
    inner = part[1]
    return inner.split('/')[0] in REMOTE_OWNED_DIRS or inner in REMOTE_OWNED_FILES


def mutable(rel):
    name = rel.rsplit('/', 1)[-1]
    return name in MUTABLE_NAMES or name.endswith('.log')


def inside(root, rel):
    """root / rel, refusing anything that resolves outside root (symbolic links included)."""
    root = Path(root).resolve()
    path = (root / clean_relative(rel)).resolve()
    if path != root and root not in path.parents:
        raise Refusal('%s resolves outside the project root' % rel)
    return path


def sha256_file(path):
    h = hashlib.sha256()
    size = 0
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(BLOCK), b''):
            h.update(block)
            size += len(block)
    return size, h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('%s.%d.tmp' % (path.name, os.getpid()))
    with open(str(tmp), 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(str(tmp), str(path))


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def try_lock(path):
    """Non-blocking exclusive lock; the open handle, or None when another process holds it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(str(path), 'a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def release(handle):
    """Unlock and close. On POSIX only close: a child that inherited the descriptor keeps the flock until it exits."""
    try:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        handle.close()


def lock_held(path):
    """True when another process holds the lock at ``path`` (an absent file is not held)."""
    path = Path(path)
    if not path.is_file():
        return False
    handle = try_lock(path)
    if handle is None:
        return True
    release(handle)
    return False


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open('/proc/%d/stat' % pid) as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


def gpu_info():
    query = 'index,name,memory.total,memory.free,memory.used,utilization.gpu,driver_version'
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in out.decode('utf-8', 'replace').strip().splitlines():
        v = [x.strip() for x in line.split(',')]
        if len(v) == 7:
            rows.append(dict(index=int(v[0]), name=v[1], memory_total_mib=float(v[2]), memory_free_mib=float(v[3]),
                             memory_used_mib=float(v[4]), utilization_pct=float(v[5]), driver=v[6]))
    return rows


def disk_free_gib(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(str(path)).free / 2 ** 30


def study_folder(root, name, frozen_ok=False):
    if not isinstance(name, str) or not STUDY_NAME.fullmatch(name):
        raise Refusal('invalid study name %r' % (name,))
    if name in FROZEN and not frozen_ok:
        raise Refusal('%s is frozen; its records are never changed' % name)
    return Path(root) / 'studies' / name


def running_queues(root):
    studies = Path(root) / 'studies'
    if not studies.is_dir():
        return []
    return sorted(p.name for p in studies.iterdir() if p.is_dir() and lock_held(p / 'worker.lock'))


def sync_summary(root):
    current = read_json(Path(root) / '.sync' / 'current.json')
    if not current:
        return None
    return dict(commit=current.get('commit'), push=current.get('push'), applied_unix=current.get('applied_unix'),
                code_files=len(current.get('code', {})), data_files=len(current.get('data', {})))


def _stdin_json():
    raw = sys.stdin.buffer.read()
    return json.loads(raw.decode('utf-8')) if raw.strip() else {}


def _walk_files(folder):
    for base, dirs, files in os.walk(str(folder), followlinks=False):
        dirs.sort()
        for name in sorted(files):
            yield Path(base) / name


# ---------------------------------------------------------------- operations

def op_ping(params):
    root = Path(params['root'])
    python = params.get('python')
    return dict(ok=True, python=sys.version.split()[0], executable=sys.executable, platform=platform.platform(),
                host=platform.node(), root=str(root), root_exists=root.is_dir(),
                root_writable=root.is_dir() and os.access(str(root), os.W_OK), disk_free_gib=disk_free_gib(root),
                research_python=python, research_python_exists=bool(python) and Path(python).is_file(),
                gpus=gpu_info(), sync=sync_summary(root), running_queues=running_queues(root))


def op_manifest(params):
    """Size and SHA-256 of files under the requested paths (stdin: {"paths": [...]}).

    skip_active: leave out runs/<job> of jobs the study's queue lists as active. require_ended: refuse unless the
    study's queue has ended (supervisor.json present, not active, supervisor process gone).
    """
    root = Path(params['root'])
    request = _stdin_json()
    study = params.get('require_ended') or params.get('skip_active')
    active_dirs, state = set(), {}
    if study:
        folder = study_folder(root, study, frozen_ok=True)
        state = read_json(folder / 'supervisor.json', {}) or {}
        if params.get('require_ended'):
            if not state:
                raise Refusal('%s has no supervisor.json: nothing has run' % study)
            if (state.get('status') in QUEUE_ACTIVE or state.get('active')
                    or (pid_alive(state.get('pid')) and lock_held(folder / 'worker.lock'))):
                raise Refusal('the queue of %s has not ended (status %s, %d active)'
                              % (study, state.get('status'), len(state.get('active') or {})))
        for value in (state.get('active') or {}).values():
            if isinstance(value, dict) and value.get('job'):
                active_dirs.add('studies/%s/runs/%s' % (study, value['job']))
    files, missing, skipped = {}, [], []

    def add(path, rel):
        reason = denied(rel)
        if reason is None and rel.endswith(TRANSIENT):
            reason = 'transient file'
        if reason is None and any(rel.startswith(a + '/') for a in active_dirs):
            reason = 'job still active'
        if reason is None and path.is_symlink():
            reason = 'symbolic link'
        if reason:
            skipped.append([rel, reason])
            return
        size, sha = sha256_file(path)
        files[rel] = dict(bytes=size, sha256=sha)

    for rel in request.get('paths', []):
        rel = clean_relative(rel)
        reason = denied(rel)
        if reason:
            skipped.append([rel, reason])
            continue
        target = inside(root, rel)
        if target.is_file():
            add(root / rel, rel)
        elif target.is_dir():
            for path in _walk_files(target):
                add(path, path.relative_to(root.resolve()).as_posix())
        else:
            missing.append(rel)
    if missing and not params.get('missing_ok'):
        raise Refusal('missing on the remote: %s' % ', '.join(missing))
    return dict(files=files, missing=missing, skipped=skipped, supervisor_status=state.get('status'))


def op_receive(params):
    """Store the push bundle from stdin as .sync/incoming/<name>.tar; returns its size and SHA-256."""
    root = Path(params['root'])
    name = params.get('name')
    if not isinstance(name, str) or not PUSH_NAME.fullmatch(name):
        raise Refusal('invalid push name %r' % (name,))
    root.mkdir(parents=True, exist_ok=True)
    incoming = root / '.sync' / 'incoming'
    incoming.mkdir(parents=True, exist_ok=True)
    for old in incoming.iterdir():  # bundles of abandoned transfers; they are never applied
        if old.is_file() and old.stat().st_mtime < time.time() - 86400:
            old.unlink()
    reserve = float(params.get('min_free_gib', 0))
    if disk_free_gib(incoming) < reserve:
        raise Refusal('less than %g GiB free on the remote disk' % reserve)
    partial, final = incoming / (name + '.tar.partial'), incoming / (name + '.tar')
    h, size = hashlib.sha256(), 0
    with open(str(partial), 'wb') as out:
        for block in iter(lambda: sys.stdin.buffer.read(BLOCK), b''):
            out.write(block)
            h.update(block)
            size += len(block)
        out.flush()
        os.fsync(out.fileno())
    os.replace(str(partial), str(final))
    return dict(name=name, bytes=size, sha256=h.hexdigest())


def op_apply(params):
    """Check and extract a received bundle, verify every manifest file on disk, record the receipt."""
    root = Path(params['root'])
    name = params.get('name')
    if not isinstance(name, str) or not PUSH_NAME.fullmatch(name):
        raise Refusal('invalid push name %r' % (name,))
    frozen = tuple(params.get('frozen', FROZEN))
    sync = root / '.sync'
    bundle = sync / 'incoming' / (name + '.tar')
    if not bundle.is_file():
        raise Refusal('bundle %s was not received' % name)
    lock = try_lock(sync / 'push.lock')
    if lock is None:
        raise Refusal('another push is being applied')
    try:
        size, sha = sha256_file(bundle)
        if sha != params.get('sha256'):
            raise Refusal('bundle SHA-256 differs from the one sent (%s)' % sha)
        if not params.get('allow_running'):
            running = running_queues(root)
            if running:
                raise Refusal('study queues are running (%s); stop them first, or pass allow_running'
                              % ', '.join(running))
        with tarfile.open(str(bundle), 'r:') as tar:
            members = tar.getmembers()
            try:
                manifest = json.loads(tar.extractfile(tar.getmember(MANIFEST_NAME)).read().decode('utf-8'))
            except KeyError:
                raise Refusal('the bundle has no %s' % MANIFEST_NAME)
            expected = dict(manifest['code'])
            expected.update(manifest['data'])
            sent = {}
            for m in members:
                if m.name == MANIFEST_NAME or m.isdir():
                    continue
                rel = clean_relative(m.name)
                if not m.isfile():
                    raise Refusal('%s: only regular files are accepted' % rel)
                reason = denied(rel) or ('written only by the remote queue' if remote_owned(rel, frozen) else None)
                if reason:
                    raise Refusal('%s: %s' % (rel, reason))
                if rel not in expected or m.size != expected[rel]['bytes']:
                    raise Refusal('%s does not match the bundle manifest' % rel)
                sent[rel] = m
            replaced = []
            for rel in manifest.get('sent_data', []):
                target = inside(root, rel)
                if target.is_file() and sha256_file(target)[1] != manifest['data'][rel]['sha256']:
                    if not params.get('replace_data'):
                        raise Refusal('%s differs on the remote; data inputs are replaced only with replace_data' % rel)
                    replaced.append(rel)
            for rel, m in sorted(sent.items()):
                target = inside(root, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.sync-partial')
                h = hashlib.sha256()
                src = tar.extractfile(m)
                with open(str(tmp), 'wb') as out:
                    for block in iter(lambda: src.read(BLOCK), b''):
                        out.write(block)
                        h.update(block)
                if h.hexdigest() != expected[rel]['sha256']:
                    tmp.unlink()
                    raise Refusal('%s: content differs from the manifest while extracting' % rel)
                if os.name == 'posix':
                    os.chmod(str(tmp), 0o755 if m.mode & 0o111 else 0o644)
                os.replace(str(tmp), str(target))
        missing, mismatches = [], []
        for rel, entry in sorted(expected.items()):
            path = root / rel
            if not path.is_file():
                missing.append(rel)
            elif sha256_file(path) != (entry['bytes'], entry['sha256']):
                mismatches.append(rel)
        previous = read_json(sync / 'current.json', {}) or {}
        stale = sorted(r for r in set(previous.get('code', {})) - set(manifest['code']) if (root / r).is_file())
        verified = not missing and not mismatches
        receipt = dict(transport_verified=verified, push=name, commit=manifest.get('commit'),
                       code_files=len(manifest['code']), data_files=len(manifest['data']), sent_files=len(sent),
                       sent_bytes=sum(m.size for m in sent.values()), missing=missing, mismatches=mismatches,
                       replaced_data=replaced, stale_files=stale, bundle_sha256=sha, bundle_bytes=size,
                       manifest_sha256=canonical_hash(manifest), applied_unix=time.time())
        write_json(sync / 'receipts' / (name + '.json'), receipt)
        if verified:
            data = dict(previous.get('data', {}))
            data.update(manifest['data'])
            write_json(sync / 'current.json', dict(commit=manifest.get('commit'), push=name,
                                                   applied_unix=receipt['applied_unix'],
                                                   manifest_sha256=receipt['manifest_sha256'],
                                                   exclude=manifest.get('exclude', []), code=manifest['code'],
                                                   data=data))
        return receipt
    finally:
        if bundle.exists():  # the local side keeps the source; a refused bundle is pushed again, not reused
            bundle.unlink()
        release(lock)


def op_send(params):
    """Tar stream of the requested files (stdin: {"files": [...]}) on stdout, after STREAM_MARK."""
    root = Path(params['root'])
    files = [clean_relative(rel) for rel in _stdin_json().get('files', [])]
    paths = []
    for rel in files:
        reason = denied(rel)
        if reason:
            raise Refusal('%s: %s' % (rel, reason))
        path = inside(root, rel)
        if not path.is_file() or (root / rel).is_symlink():
            raise Refusal('%s is not a regular file on the remote' % rel)
        paths.append((rel, path))
    out = sys.stdout.buffer
    out.write(STREAM_MARK)
    with tarfile.open(fileobj=out, mode='w|', format=tarfile.PAX_FORMAT) as tar:
        for rel, path in paths:
            info = tar.gettarinfo(str(path), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            with open(str(path), 'rb') as f:
                tar.addfile(info, f)
    out.flush()
    return None


def op_status(params):
    root = Path(params['root'])
    study = params.get('study')
    folder = study_folder(root, study, frozen_ok=True)
    supervisor = read_json(folder / 'supervisor.json')
    tail = []
    events = folder / 'events.jsonl'
    if events.is_file():
        size = events.stat().st_size
        with open(str(events), 'rb') as f:
            f.seek(max(0, size - 65536))
            lines = f.read().decode('utf-8', 'replace').splitlines()
        if size > 65536:
            lines = lines[1:]  # the first line may be cut
        for line in lines[-int(params.get('events', 20)):]:
            try:
                tail.append(json.loads(line))
            except ValueError:
                pass
    stop = folder / 'STOP'
    runs = folder / 'runs'
    pid = (supervisor or {}).get('pid')
    return dict(study=study, exists=folder.is_dir(), supervisor=supervisor,
                supervisor_alive=pid_alive(pid) if isinstance(pid, int) else False, stop=stop.is_file(),
                stop_record=stop.read_text(encoding='utf-8', errors='replace')[:2000] if stop.is_file() else None,
                worker_lock='held' if lock_held(folder / 'worker.lock') else 'free',
                runs=sum(1 for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else 0, events_tail=tail,
                gpus=gpu_info(), disk_free_gib=disk_free_gib(root), sync=sync_summary(root))


def op_stop(params):
    """Create studies/<name>/STOP (the frozen study's STOP format); an existing STOP is left as it is."""
    folder = study_folder(params['root'], params.get('study'))
    if not folder.is_dir():
        raise Refusal('%s does not exist on the remote' % folder)
    stop = folder / 'STOP'
    if stop.exists():
        return dict(created=False, path=str(stop), record=stop.read_text(encoding='utf-8', errors='replace')[:2000])
    record = dict(reason=params.get('reason') or 'user_requested_pause',
                  requested_at=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                  resume_requires_explicit_user_request=True, requested_by='tools.remote stop')
    fd = os.open(str(stop), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(json.dumps(record, indent=2) + '\n')
    return dict(created=True, path=str(stop), record=record)


def op_launch(params):
    """Start research.py --study <name> run on the remote, detached, logging to studies/<name>/queue.log."""
    root = Path(params['root'])
    study = params.get('study')
    folder = study_folder(root, study)
    if not folder.is_dir() or not (root / 'research.py').is_file():
        raise Refusal('%s or research.py is missing on the remote; push first' % folder)
    python = params.get('python')
    if not python or not Path(python).is_file():
        raise Refusal('the research interpreter %s is missing; set up the remote environment first' % python)
    args = params.get('args', [])
    if not all(isinstance(a, str) and SAFE_ARG.fullmatch(a) for a in args):
        raise Refusal('invalid queue arguments %r' % (args,))
    stop = folder / 'STOP'
    cleared = None
    if stop.exists():
        if not params.get('clear_stop'):
            raise Refusal('%s exists; pass clear_stop for an explicit restart' % stop)
        cleared = folder / ('STOP-cleared-%s.json' % time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()))
        os.replace(str(stop), str(cleared))
    if lock_held(folder / 'worker.lock'):
        raise Refusal('the queue of %s is already running' % study)
    command = [python, '-B', '-X', 'utf8', '-u', str(root / 'research.py'), '--study', study, 'run'] + list(args)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', LANG='C',
               LC_ALL='C', PYTHONPATH=str(root / 'flybrain'))
    log_path = folder / 'queue.log'
    kwargs = dict(cwd=str(root), env=env, stdin=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    if os.name == 'posix':
        kwargs['start_new_session'] = True
    else:
        kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    with open(str(log_path), 'ab') as log:
        process = subprocess.Popen(command, stdout=log, **kwargs)
    time.sleep(float(params.get('settle_seconds', 2)))
    code = process.poll()
    tail = None
    if code is not None:
        with open(str(log_path), 'rb') as f:
            f.seek(max(0, log_path.stat().st_size - 4000))
            tail = f.read().decode('utf-8', 'replace')
    return dict(pid=process.pid, command=command, log=str(log_path), exit_code=code, log_tail=tail,
                stop_cleared=str(cleared) if cleared else None)


OPS = dict(ping=op_ping, manifest=op_manifest, receive=op_receive, apply=op_apply, send=op_send, status=op_status,
           stop=op_stop, launch=op_launch)


def main(argv):
    params = json.loads(base64.b64decode(argv[-1]).decode('utf-8'))
    op = OPS.get(params.get('op'))
    if op is None:
        sys.stderr.write(json.dumps(dict(refused='unknown operation %r' % params.get('op'))) + '\n')
        return REFUSED
    try:
        result = op(params)
    except Refusal as refusal:
        sys.stderr.write(json.dumps(dict(refused=str(refusal))) + '\n')
        return REFUSED
    if result is not None:
        sys.stdout.write(RESULT_MARK + json.dumps(result) + '\n')
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
