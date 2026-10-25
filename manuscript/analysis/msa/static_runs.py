"""Loaders for the 68 local static runs (difficulty ladder, training controls, benchmarks)."""
from functools import lru_cache

from . import io, paths


def run_ids(prefix=None):
    ids = sorted(p.name for p in paths.RUNS.iterdir() if p.is_dir())
    return [r for r in ids if prefix is None or r.startswith(prefix)]


def run_dir(run_id):
    return paths.RUNS / run_id


def _opt(run_id, name):
    p = run_dir(run_id) / name
    return io.json_load(p) if p.exists() else None


@lru_cache(maxsize=None)
def config(run_id):
    return io.json_load(run_dir(run_id) / 'config.json')


@lru_cache(maxsize=None)
def result(run_id):
    return _opt(run_id, 'result.json')


@lru_cache(maxsize=None)
def progress(run_id):
    return _opt(run_id, 'progress.json')


@lru_cache(maxsize=None)
def verification(run_id):
    return _opt(run_id, 'verification.json')


@lru_cache(maxsize=None)
def acceptance(run_id):
    return _opt(run_id, 'acceptance.json')


@lru_cache(maxsize=None)
def environment(run_id):
    return _opt(run_id, 'environment.json')


@lru_cache(maxsize=None)
def brain(run_id):
    return _opt(run_id, 'brain.json')


@lru_cache(maxsize=None)
def before(run_id):
    return _opt(run_id, 'before.json')


@lru_cache(maxsize=None)
def after(run_id):
    return _opt(run_id, 'after.json')


@lru_cache(maxsize=None)
def history(run_id):
    p = run_dir(run_id) / 'history.jsonl'
    return io.jsonl_load(p) if p.exists() else []


@lru_cache(maxsize=None)
def exposure(run_id):
    p = run_dir(run_id) / 'exposure.jsonl'
    return io.jsonl_load(p) if p.exists() else []


@lru_cache(maxsize=None)
def behavior(run_id):
    p = run_dir(run_id) / 'behavior.jsonl'
    return io.jsonl_load(p) if p.exists() else []


@lru_cache(maxsize=None)
def gradients(run_id):
    p = run_dir(run_id) / 'gradients.jsonl'
    return io.jsonl_load(p) if p.exists() else []


@lru_cache(maxsize=None)
def extensions(run_id):
    p = run_dir(run_id) / 'extensions.jsonl'
    return io.jsonl_load(p) if p.exists() else []


def interruptions(run_id):
    return [io.json_load(p) for p in sorted(run_dir(run_id).glob('interruption-*/interruption.json'))]


def status(run_id):
    """Recorded end status: result.json, else progress.json."""
    r = result(run_id)
    if r and r.get('status'):
        return r['status']
    p = progress(run_id)
    return p.get('status') if p else None


def checkpoint(run_id, name='model.pt'):
    p = run_dir(run_id) / name
    return io.torch_load(p) if p.exists() else None
