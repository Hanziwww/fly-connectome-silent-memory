"""Active results, immutable historical records, and local runtime locations."""
import hashlib
import json
from pathlib import Path
import zipfile

from bootstrap import HERE, ROOT

ARCHIVE = ROOT / 'archive/experiments'
RUNS = ROOT / 'runs'
RUNTIME = ROOT / 'runtime'
MAINTENANCE = ROOT / 'archive/maintenance/20261002'
METHOD = json.loads((ROOT / 'method.json').read_text(encoding='utf-8'))
CURRENT_RUN = ROOT / METHOD['current_run']
EVIDENCE = {
    'p03': RUNS / 'linear-joint/20261002-p03-01/verification.json',
    'p05': RUNS / 'linear-joint/20261002-p05-01/verification.json',
    'repair': ARCHIVE / 'repair/20261001-b00-intervention01/verification.json',
    'pilot': ARCHIVE / 'campaign/run/viewer-verification.json',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def recorded_path(value):
    """Resolve preserved pre-organization paths without rewriting evidence files."""
    path = Path(value)
    if path.exists():
        return path
    manifest = read(MAINTENANCE / 'relocation-manifest.json')
    if path.is_absolute():
        for base in (ROOT, Path(manifest['original_root'])):
            try:
                relative = path.relative_to(base).as_posix()
                break
            except ValueError:
                pass
        else:
            return path
    else:
        relative = path.as_posix()
    for item in manifest['moves']:
        old = item['source']
        if relative == old or relative.startswith(old + '/'):
            return ROOT / (item['destination'] + relative[len(old):])
    return ROOT / relative


def verify_code_manifest(directory):
    """Old runs use their frozen source snapshot; new runs check the working code."""
    directory = Path(directory).resolve()
    expected = read(directory / 'source-files.json')
    frozen = {ROOT / item['run'] for item in METHOD['retained_results']}
    if directory in frozen:
        receipt = read(MAINTENANCE / 'relocation-manifest.json')
        with zipfile.ZipFile(ROOT / receipt['snapshot']) as archive:
            for name, digest in expected.items():
                assert hashlib.sha256(archive.read(name.replace('\\', '/'))).hexdigest() == digest, name
    else:
        for name, digest in expected.items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def source_manifest():
    paths = list(HERE.glob('*.py')) + list((HERE / 'lib').glob('*.py')) + [ROOT / 'workflow.py', ROOT / 'method.json']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
