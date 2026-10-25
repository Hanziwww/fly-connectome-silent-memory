"""Read-only hash verification of a study's local mirror against the remote; generalises tools/verify_remote_sync.py.

The study's remote queue must have ended (no active worker). The remote lists every output file of the study
(runs/, code-snapshots/, supervisor.json, events.jsonl, queue.log) with size and SHA-256, and each must exist
locally with the same size and hash. The result keeps the JSON schema of tools/verify_remote_sync.py:

    transport_verified, scientific_acceptance_separate, files, bytes, missing, mismatches, manifest

with project-relative paths as manifest keys, written to studies/<name>/sync-verification.json. Transport
verification says nothing about scientific acceptance; analyses use only files listed in a verification whose
transport_verified is true (``verified_manifest``).
"""
import json
from pathlib import Path

from . import ROOT, RemoteError, agent
from .sync import check_pull_paths, check_study, study_outputs


def compare(manifest, root=ROOT):
    """Check local files against a remote manifest {relative path: {bytes, sha256}}."""
    root = Path(root)
    missing, mismatches = [], []
    for name, expected in manifest.items():
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        if path.stat().st_size != expected['bytes'] or agent.sha256_file(path)[1] != expected['sha256']:
            mismatches.append(name)
    return dict(transport_verified=not missing and not mismatches, scientific_acceptance_separate=True,
                files=len(manifest), bytes=sum(v['bytes'] for v in manifest.values()), missing=missing,
                mismatches=mismatches, manifest=manifest)


def verify(ssh, study, paths=None, root=ROOT, out=None):
    """Verify the local mirror of ``study`` against the remote; writes and returns the verification record."""
    check_study(study)
    rels = check_pull_paths(study, paths or study_outputs(study))
    listing = ssh.call('manifest', input=dict(paths=rels), missing_ok=True, require_ended=study)
    result = compare(listing['files'], root)
    target = Path(out) if out else Path(root) / 'studies' / study / 'sync-verification.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    return result


def verified_manifest(study, root=ROOT, require=()):
    """Manifest of the study's last verification; raises unless transport_verified and every ``require`` path is in it."""
    path = Path(root) / 'studies' / study / 'sync-verification.json'
    if not path.is_file():
        raise RemoteError(f'{study}: no sync-verification.json; run python -m tools.remote verify --study {study}')
    record = json.loads(path.read_text(encoding='utf-8'))
    if not record.get('transport_verified'):
        raise RemoteError(f'{study}: the last transport verification failed (missing {len(record["missing"])}, '
                          f'mismatched {len(record["mismatches"])})')
    absent = [p for p in require if p not in record['manifest']]
    if absent:
        raise RemoteError(f'{study}: not covered by the transport verification: {absent}')
    return record['manifest']
