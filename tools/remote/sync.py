"""Push code and data to the remote project mirror and pull study outputs back, SHA-256 verified both ways.

push  ``git archive <commit>`` (minus the remote.toml ``exclude`` pathspecs) plus listed data files, streamed over
      ssh as one tar with a manifest. The remote stores the bundle, checks its hash, refuses while any study queue
      holds its lock, extracts it with path checks, verifies every manifest file on disk and records a receipt
      (.sync/receipts/<push>.json, .sync/current.json). A data file is sent only when the remote copy is missing or
      different; a different remote copy is replaced only with replace_data. Receipt copy: runtime/remote/pushes/.
pull  remote manifest (path, size, SHA-256) of a study's outputs -> tar of the files missing or different locally ->
      each file checked against the manifest before it replaces anything. Evidence is never overwritten: a local
      file that differs from the remote is a conflict, except mirrored state (supervisor.json, events.jsonl,
      progress.json, *.log). Jobs the remote queue lists as active are skipped. Receipt: studies/<name>/receipts/.

Never transferred: local/, runtime/, .git/, .sync/, environments, remote.toml, ssh.txt, keys and known_hosts files
(agent.denied). The frozen study is never a pull destination and its remote copy is only ever pushed code.
"""
import datetime
import hashlib
import io
import json
import os
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from . import ROOT, RemoteError, agent
from .ssh import AGENT_SHA256

FROZEN_STUDIES = agent.FROZEN
STUDY_FILES = ('protocol.json', 'schedule.json', 'tasks/lock.json')
STUDY_OUTPUTS = ('runs', 'code-snapshots', 'supervisor.json', 'events.jsonl', 'queue.log')


def _stamp():
    # Microseconds: parallel pulls of one study must not write the same receipt name.
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def _rel(text):
    try:
        return agent.clean_relative(text)
    except agent.Refusal as refusal:
        raise RemoteError(str(refusal)) from None


def _inside(root, rel):
    try:
        return agent.inside(root, rel)
    except agent.Refusal as refusal:
        raise RemoteError(str(refusal)) from None


def _git(root, *args, git='git'):
    result = subprocess.run([git, '-C', str(root), *args], capture_output=True)
    if result.returncode:
        raise RemoteError(f'git {" ".join(args)} failed: {result.stderr.decode("utf-8", "replace").strip()}')
    return result.stdout.decode('utf-8', 'replace')


def resolve_commit(root=ROOT, commit='HEAD', git='git'):
    try:
        return _git(root, 'rev-parse', '--verify', '--quiet', f'{commit}^{{commit}}', git=git).strip()
    except RemoteError:
        raise RemoteError(f'{commit!r} is not a commit of {root}') from None


def check_study(study, frozen_ok=False):
    if not isinstance(study, str) or not agent.STUDY_NAME.fullmatch(study):
        raise RemoteError(f'invalid study name {study!r}')
    if study in FROZEN_STUDIES and not frozen_ok:
        raise RemoteError(f'{study} is frozen; its records are never changed')
    return study


def dirty_files(root=ROOT, git='git'):
    """Tracked files with uncommitted changes (they are not part of a push of HEAD)."""
    return [line[3:] for line in _git(root, 'status', '--porcelain', '--untracked-files=no', git=git).splitlines()]


def committed_files(root, sha, prefix, git='git'):
    names = _git(root, 'ls-tree', '-r', '-z', '--name-only', sha, '--', prefix, git=git).split('\0')
    return {p for p in names if p}


def code_files(root, sha, exclude, git='git'):
    """Files a code push of ``sha`` carries: the commit's files outside the ``exclude`` paths."""
    def excluded(path):
        return any(path == e or path.startswith(e.rstrip('/') + '/') for e in exclude)
    names = _git(root, 'ls-tree', '-r', '-z', '--name-only', sha, git=git).split('\0')
    return {p for p in names if p and not excluded(p)}


def read_list(path):
    """Paths from a transfer-list file: one project-relative path per line; '#' starts a comment."""
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    return [line.split('#', 1)[0].strip() for line in lines if line.split('#', 1)[0].strip()]


def expand_paths(root, paths):
    """{relative path: file} for listed files, directories (recursive) and glob patterns, checked for transfer."""
    root = Path(root).resolve()
    found = {}
    for item in paths:
        rel = _rel(str(item).replace('\\', '/'))
        reason = agent.denied(rel)
        if reason:
            raise RemoteError(f'{rel}: {reason}')
        matches = sorted(root.glob(rel)) if any(c in rel for c in '*?[') else [root / rel]
        if not matches or not matches[0].exists():
            raise RemoteError(f'{rel} does not exist')
        for match in matches:
            files = [match] if match.is_file() else sorted(p for p in match.rglob('*') if p.is_file())
            for path in files:
                try:
                    sub = path.resolve().relative_to(root).as_posix()
                except ValueError:
                    raise RemoteError(f'{path} resolves outside the project') from None
                reason = agent.denied(sub) or ('written only by the remote queue' if agent.remote_owned(sub) else None)
                if reason:
                    raise RemoteError(f'{sub}: {reason}')
                if not sub.endswith(agent.TRANSIENT) and '__pycache__' not in sub.split('/'):
                    found[sub] = path
    return found


def hash_files(files):
    return {rel: dict(zip(('bytes', 'sha256'), agent.sha256_file(path))) for rel, path in sorted(files.items())}


class _HashingWriter:
    def __init__(self, target):
        self.target, self.hash, self.bytes = target, hashlib.sha256(), 0

    def write(self, data):
        self.target.write(data)
        self.hash.update(data)
        self.bytes += len(data)
        return len(data)

    def flush(self):
        self.target.flush()


class _HashingReader:
    def __init__(self, source):
        self.source, self.hash, self.bytes = source, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.hash.update(data)
        self.bytes += len(data)
        return data


def _clean_info(info, rel):
    out = tarfile.TarInfo(rel)
    out.size, out.mtime, out.type = info.size, int(info.mtime), tarfile.REGTYPE
    out.mode = 0o755 if info.mode & 0o111 else 0o644
    return out


def _archive_members(root, sha, exclude, git='git'):
    """(TarInfo, file object) for every regular file of ``git archive <sha>``, read as a stream."""
    spec = ['--', '.', *[f':(exclude){e}' for e in exclude]] if exclude else []
    process = subprocess.Popen([git, '-C', str(root), 'archive', '--format=tar', sha, *spec],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    drain = threading.Thread(target=lambda: err.append(process.stderr.read()), daemon=True)
    drain.start()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as tar:
            for info in tar:
                if info.isdir():
                    continue
                if not info.isfile():
                    raise RemoteError(f'{info.name}: the commit holds a link or special file; only regular files '
                                      'are transferred')
                yield info, tar.extractfile(info)
        # tarfile stops at the end-of-archive marker, but git pads the archive to a whole record; read the rest so git
        # can finish writing (closing the pipe early made git fail with an empty message, depending on the padding)
        while process.stdout.read(1 << 16):
            pass
    finally:
        process.stdout.close()
        code = process.wait()
        drain.join()
    if code:
        raise RemoteError(f'git archive failed: {b"".join(err).decode("utf-8", "replace").strip()}')


def push(ssh, commit='HEAD', data=(), study=None, root=ROOT, replace_data=False, allow_running=False, dry_run=False,
         git='git', record=True):
    """Push ``commit`` (minus config.exclude) and the ``data`` paths; returns the remote receipt."""
    root = Path(root).resolve()
    config = ssh.config
    sha = resolve_commit(root, commit, git)
    files = expand_paths(root, data)
    if study:
        folder = f'studies/{check_study(study, frozen_ok=True)}'
        present = committed_files(root, sha, folder, git)
        absent = [f'{folder}/{name}' for name in STUDY_FILES if f'{folder}/{name}' not in present
                  and f'{folder}/{name}' not in files]
        if absent:
            raise RemoteError(f'commit {sha[:12]} lacks {", ".join(absent)} (untracked? .gitignore whitelists JSON '
                              'only for brain-sokoban-v1); commit them or pass them as data paths')
    expected_code = code_files(root, sha, config.exclude, git)
    overlap = sorted(set(files) & expected_code)
    if overlap:
        raise RemoteError(f'already part of the code push of {sha[:12]}, so not data: {overlap[:5]}')
    local = hash_files(files)
    remote = ssh.call('manifest', input=dict(paths=sorted(local)), missing_ok=True)['files'] if local else {}
    send = [rel for rel in sorted(local) if remote.get(rel) != local[rel]]
    differing = [rel for rel in send if rel in remote]
    if differing and not replace_data:
        raise RemoteError(f'data files differ on the remote: {differing}; pass replace_data to replace them')
    summary = dict(commit=sha, exclude=list(config.exclude), data_files=len(local), data_to_send=send,
                   data_bytes_to_send=sum(local[r]['bytes'] for r in send), data_replaced=differing,
                   dirty_tracked_files=dirty_files(root, git) if commit == 'HEAD' else [])
    if dry_run:
        return dict(summary, dry_run=True)
    name = f'{_stamp()}-{sha[:12]}'
    state = {}

    def write_bundle(stream):
        out_hash = _HashingWriter(stream)
        with tarfile.open(fileobj=out_hash, mode='w|', format=tarfile.PAX_FORMAT) as out:
            code = {}
            for info, source in _archive_members(root, sha, config.exclude, git):
                rel = _rel(info.name)
                reason = agent.denied(rel) or ('written only by the remote queue' if agent.remote_owned(rel) else None)
                if reason:
                    raise RemoteError(f'commit {sha[:12]} contains {rel} ({reason}); it is never transferred')
                reader = _HashingReader(source)
                out.addfile(_clean_info(info, rel), reader)
                code[rel] = dict(bytes=info.size, sha256=reader.hash.hexdigest())
            if set(code) != expected_code:
                raise RemoteError(f'git archive of {sha[:12]} differs from its file list: '
                                  f'{sorted(set(code) ^ expected_code)[:5]}')
            for rel in send:
                with files[rel].open('rb') as f:
                    reader = _HashingReader(f)
                    info = tarfile.TarInfo(rel)
                    info.size, info.mtime, info.mode = local[rel]['bytes'], int(files[rel].stat().st_mtime), 0o644
                    out.addfile(info, reader)
                if reader.hash.hexdigest() != local[rel]['sha256']:
                    raise RemoteError(f'{rel} changed while it was being sent')
            manifest = dict(schema_version=1, commit=sha, push=name, exclude=list(config.exclude), code=code,
                            data=local, sent_data=send, created_unix=time.time(), agent_sha256=AGENT_SHA256)
            raw = json.dumps(manifest, sort_keys=True).encode('utf-8')
            info = tarfile.TarInfo(agent.MANIFEST_NAME)
            info.size, info.mtime, info.mode = len(raw), int(time.time()), 0o644
            out.addfile(info, io.BytesIO(raw))
        state.update(sha256=out_hash.hash.hexdigest(), bytes=out_hash.bytes, manifest=manifest)

    received = ssh.upload('receive', write_bundle, name=name, min_free_gib=config.min_free_disk_gib)
    if (received['sha256'], received['bytes']) != (state['sha256'], state['bytes']):
        raise RemoteError(f'the remote received {received["bytes"]} bytes with SHA-256 {received["sha256"]}, '
                          f'not the {state["bytes"]} bytes sent ({state["sha256"]})')
    receipt = ssh.call('apply', name=name, sha256=state['sha256'], replace_data=replace_data,
                       allow_running=allow_running, frozen=list(FROZEN_STUDIES))
    if receipt['manifest_sha256'] != agent.canonical_hash(state['manifest']):
        raise RemoteError('the remote applied a different manifest than the one sent')
    receipt = dict(receipt, host=config.host, remote_root=config.remote_root, **summary)
    if record:
        agent.write_json(root / 'runtime' / 'remote' / 'pushes' / f'{name}.json', receipt)
    if not receipt['transport_verified']:
        raise RemoteError(f'push {name} did not verify: missing {receipt["missing"]}, mismatches '
                          f'{receipt["mismatches"]}')
    return receipt


def check_pull_paths(study, paths):
    """Pull targets: remote-owned outputs inside studies/<study>/ only (never the frozen study)."""
    if study in FROZEN_STUDIES:
        raise RemoteError(f'{study} is frozen; it is never a pull destination')
    check_study(study)
    out = []
    for rel in paths:
        rel = _rel(str(rel).replace('\\', '/'))
        part = agent.study_part(rel)
        if part is None or part[0] != study or not agent.remote_owned(rel):
            raise RemoteError(f'{rel}: a pull of {study} brings only its remote outputs '
                              f'(studies/{study}/{{{",".join(STUDY_OUTPUTS)}}})')
        out.append(rel)
    return out


def study_outputs(study):
    return [f'studies/{study}/{name}' for name in STUDY_OUTPUTS]


def pull(ssh, study, paths=None, root=ROOT, dry_run=False):
    """Fetch the remote outputs of ``study`` that are missing or out of date locally; returns the receipt."""
    root = Path(root).resolve()
    rels = check_pull_paths(study, paths or study_outputs(study))
    listing = ssh.call('manifest', input=dict(paths=rels), missing_ok=True, skip_active=study)
    remote = listing['files']
    fetch, update, present, conflicts = [], [], [], []
    for rel, entry in sorted(remote.items()):
        if not agent.remote_owned(rel):
            raise RemoteError(f'the remote listed {rel}, outside the requested outputs')
        path = _inside(root, rel)
        if not path.is_file():
            fetch.append(rel)
        elif agent.sha256_file(path) == (entry['bytes'], entry['sha256']):
            present.append(rel)
        elif agent.mutable(rel):
            update.append(rel)
        else:
            conflicts.append(rel)
    plan = dict(study=study, files=len(remote), fetch=fetch, update=update, present=len(present),
                conflicts=conflicts, skipped=listing['skipped'], remote_missing=listing['missing'])
    if dry_run:
        return dict(plan, dry_run=True)
    wanted = set(fetch) | set(update)
    received, mismatches = [], []

    def reader(stream):
        with tarfile.open(fileobj=stream, mode='r|') as tar:
            for info in tar:
                rel = _rel(info.name)
                if rel not in wanted or not info.isfile():
                    raise RemoteError(f'unexpected member {info.name!r} in the pulled stream')
                target = _inside(root, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.download')
                hashed = _HashingReader(tar.extractfile(info))
                with tmp.open('wb') as out:
                    for block in iter(lambda: hashed.read(1 << 20), b''):
                        out.write(block)
                if (hashed.bytes, hashed.hash.hexdigest()) == (remote[rel]['bytes'], remote[rel]['sha256']):
                    os.replace(tmp, target)
                    received.append(rel)
                else:
                    tmp.unlink()
                    mismatches.append(rel)

    if wanted:
        ssh.download('send', reader, input=dict(files=sorted(wanted)))
    missing = sorted(wanted - set(received) - set(mismatches))
    receipt = dict(transport_verified=not (missing or mismatches or conflicts), study=study, host=ssh.config.host,
                   remote_root=ssh.config.remote_root, paths=rels, files=len(remote),
                   bytes=sum(v['bytes'] for v in remote.values()), fetched=sorted(set(fetch) & set(received)),
                   updated=sorted(set(update) & set(received)), present=len(present), conflicts=conflicts,
                   missing=missing, mismatches=sorted(mismatches), skipped=listing['skipped'],
                   manifest_sha256=agent.canonical_hash(remote), pulled_unix=time.time())
    agent.write_json(root / 'studies' / study / 'receipts' / f'pull-{_stamp()}.json', receipt)
    return receipt
