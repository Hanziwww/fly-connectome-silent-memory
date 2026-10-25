"""Read-only guard: an audit hook that blocks writes outside the two writable roots.

Installed once per process before any input is read. It also hides CUDA devices so no
code path can reach a GPU.
"""
import os
import sys
import tempfile
from pathlib import Path

from . import paths

_installed = False
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_PATH_EVENTS = {'os.remove', 'os.rmdir', 'os.mkdir', 'os.truncate', 'os.chmod',
                'os.utime', 'os.symlink', 'os.link', 'os.chown', 'shutil.rmtree'}
# Resolved once, before the hook exists: tempfile.gettempdir() probes the disk under a lock
# and would re-enter the hook if it were called from inside it.
_ROOTS = []


def _resolve_roots():
    return [Path(r).resolve() for r in paths.WRITABLE] + [Path(tempfile.gettempdir()).resolve()]


def allowed(path):
    roots = _ROOTS or _resolve_roots()
    try:
        target = Path(os.fsdecode(path)).resolve()
    except (TypeError, ValueError, OSError):
        return False
    return any(target == root or root in target.parents for root in roots)


def _hook(event, args):
    if event == 'open':
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        if isinstance(mode, str):
            writing = any(c in mode for c in 'wax+')
        else:
            writing = isinstance(flags, int) and bool(flags & _WRITE_FLAGS)
        if writing and not allowed(path):
            raise PermissionError(f'read-only guard blocked a write to {path}')
    elif event in _PATH_EVENTS:
        path = args[0] if args else None
        if path is None or isinstance(path, int):
            return
        if not allowed(path):
            raise PermissionError(f'read-only guard blocked {event} on {path}')
    elif event == 'os.rename':
        if not (allowed(args[0]) and allowed(args[1])):
            raise PermissionError(f'read-only guard blocked a rename {args[0]} -> {args[1]}')


def install():
    global _installed
    if _installed:
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    _ROOTS.extend(_resolve_roots())
    sys.addaudithook(_hook)
    _installed = True
