"""Input roots (read-only) and the only two writable locations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'studies' / 'brain-sokoban-v1'
RUNS = STUDY / 'runs'
TASKS = STUDY / 'tasks'
CODE = STUDY / 'code'
M1 = STUDY / 'mechanisms-v1'
MS = STUDY / 'mechanisms-supplement-v1'
R0_ROOT = STUDY / 'remote-20261003'
R0 = R0_ROOT / 'results'
RS_ROOT = STUDY / 'remote-supplement-20261003'
RS = RS_ROOT / 'results'
LINEAGE = ROOT / 'runs' / 'linear-joint'
ARCHIVE = ROOT / 'archive'
DOCS = ROOT / 'docs'
CONNECTOME = ROOT / 'flybrain' / 'cache'
GAME = ROOT / 'sokoban' / 'lib'

ANALYSIS = ROOT / 'manuscript' / 'analysis'
OUT = ROOT / 'manuscript' / 'results'
WORK = ANALYSIS / '_work'
WRITABLE = (OUT, WORK)


def rel(path):
    """Project-relative POSIX path used in every ``source_file`` column."""
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def _inside(root, relpath):
    root = root.resolve()
    target = (root / relpath).resolve()
    if target != root and root not in target.parents:
        raise PermissionError(f'{relpath} escapes {root}')
    return target


def out(relpath):
    """Location of a deliverable under manuscript/results."""
    return _inside(OUT, relpath)


def work(relpath):
    """Location of an intermediate file under manuscript/analysis/_work."""
    return _inside(WORK, relpath)
