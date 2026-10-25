"""Registered studies of plan v2:  research.py --study NAME {status,run,verify} [options]

status  read-only summary of the study: protocol, registration, schedule, task lock, amendments, test plan, STOP,
        worker lock and queue state. It writes nothing, so it is always safe (a dry run).
run     the study's supervisor: checks the registered records, takes the per-study worker lock
        (studies/NAME/worker.lock) and calls run(study, args) in studies/NAME/code/study.py, which normally runs the
        lane/slot queue (tools/remote/queue.py). With --job ID it runs one job in this process instead:
        run_job(study, job, args).
verify  the same checks, then verify(study, args), or verify_job(study, job, args) with --job; a job's verification
        runs in its own process, separate from the one that produced its result.

--dry-run (run, verify) does the checks and reports what would be dispatched, without the lock or the dispatch.
Exit codes: 0 done, 1 error, 2 refused (draft or changed records, frozen study, lock held, STOP, not implemented),
75 stopped at a safe point because of STOP. The frozen study brain-sokoban-v1 keeps its own entry point
(workflow.py study); here it is read-only (status only).

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script research.py --study NAME status
"""
import argparse
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studykit.registry import (EXIT_STOPPED, Study, StudyError, StudyLocked, StudyStopped, digest,  # noqa: E402
                               file_lock)

REFUSED = 2


class Refused(Exception):
    """The command is not allowed in the study's current state."""


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_name(study, kind):
    return f'studycode_{study.name.replace("-", "_")}_{kind}'


def study_code(study):
    """studies/NAME/code/study.py, imported with its folder on sys.path (for the study's sibling modules)."""
    entry = study.code_dir / 'study.py'
    if not entry.is_file():
        raise Refused(f'{study.relative(entry)} is missing')
    if str(study.code_dir) not in sys.path:
        sys.path.insert(0, str(study.code_dir))
    return _load(entry, _module_name(study, 'study'))


def test_plan_summary(study):
    path = study.code_dir / 'testplan.py'
    if not path.is_file():
        return None
    try:
        module = _load(path, _module_name(study, 'testplan'))
    except Exception as error:  # a broken declaration module is reported, not fatal, in a status
        return dict(error=repr(error))
    plan = getattr(module, 'PLAN', None)
    if plan is not None and hasattr(plan, 'summary'):
        return dict(plan.summary(), sha256=plan.digest())
    families = getattr(module, 'FAMILIES', None)
    return dict(families=len(families)) if families is not None else dict(error='no PLAN or FAMILIES declared')


def _guard(fn):
    try:
        return fn()
    except (StudyError, ValueError, OSError) as error:
        return dict(error=str(error))


def _events(study):
    path = study.events_path
    if not path.is_file():
        return dict(count=0, last=None)
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    try:
        last = json.loads(lines[-1]) if lines else None
    except ValueError:
        last = dict(error='last line is not JSON')
    return dict(count=len(lines), last=last)


def status(study):
    """Everything is read; nothing is created, locked or written."""
    out = dict(study=study.name, path=study.relative(study.path), exists=study.path.is_dir(), frozen=study.frozen,
               dry_run=True)
    if not out['exists']:
        return out

    def protocol():
        value = study.load_protocol()
        return dict(status=value['status'], schema_version=value['schema_version'], sha256=digest(study.protocol_path))

    def schedule():
        return dict(jobs=len(study.schedule()), sha256=digest(study.schedule_path))

    def amendments():
        chain = study.amendments()
        return dict(count=len(chain), head_sha256=chain[-1]['sha256'] if chain else None)

    def queue():
        from tools.remote import queue as remote_queue
        return remote_queue.status(study)

    frozen = 'not applicable: frozen study (its own records)'
    out.update(protocol=_guard(protocol),
               registration=frozen if study.frozen else _guard(study.check_registration),
               schedule=_guard(schedule), task_lock=_guard(study.check_task_lock),
               amendments=frozen if study.frozen else _guard(amendments),
               test_plan=None if study.frozen else test_plan_summary(study), queue=_guard(queue),
               events=_guard(lambda: _events(study)),
               runs=sum(1 for p in study.runs_dir.iterdir() if p.is_dir()) if study.runs_dir.is_dir() else 0,
               code_snapshots=(sum(1 for p in study.snapshots_dir.iterdir() if p.is_dir())
                               if study.snapshots_dir.is_dir() else 0),
               code_entry=(study.code_dir / 'study.py').is_file())
    return out


def preflight(study):
    """Refuse unless the study is registered, unchanged since registration, not stopped and not frozen."""
    if study.frozen:
        raise Refused(f'{study.name} is frozen; it is read-only here (its own entry point is workflow.py study)')
    if not study.path.is_dir():
        raise Refused(f'{study.relative(study.path)} does not exist')
    try:
        protocol = study.load_protocol()
        if protocol['status'] != 'registered':
            raise Refused(f"the protocol is {protocol['status']!r}; run and verify need a registered protocol")
        registration = study.check_registration()
        if not registration['ok']:
            raise Refused(f"records changed since registration: {registration.get('changed') or registration}")
        lock = study.check_task_lock()
        if not lock['ok']:
            raise Refused(f'tasks/ differs from tasks/lock.json: {lock}')
        study.amendments()
        jobs = study.schedule()
    except StudyError as error:
        raise Refused(str(error)) from None
    if study.stopped():
        raise Refused(f'{study.relative(study.stop_path)} exists; remove it only for an explicit restart')
    return jobs


def dispatch(study, args):
    jobs = preflight(study)
    verb = args.command
    if args.job is not None:
        job = next((j for j in jobs if j['id'] == args.job), None)
        if job is None:
            raise Refused(f'{args.job!r} is not a job of the schedule')
        name = f'{verb}_job'
    else:
        job, name = None, verb
    if args.dry_run:
        return dict(study=study.name, dry_run=True, would_call=f'studies/{study.name}/code/study.py:{name}',
                    job=args.job, jobs=len(jobs), worker_lock=study.lock_state()), 0
    function = getattr(study_code(study), name, None)
    if function is None:
        raise Refused(f'studies/{study.name}/code/study.py defines no {name}()')
    if job is None:
        lock = study.worker_lock()
    else:
        # A queue worker inherits the study lock from its supervisor; a job started by hand takes it itself.
        lock = contextlib.nullcontext() if study.held_by_parent() else study.worker_lock()
    try:
        with lock, (file_lock(study.run_dir(job['id']) / 'job.lock') if job else contextlib.nullcontext()):
            study.event(f'{verb}_started', job=args.job, pid=os.getpid())
            code = function(study, job, args) if job else function(study, args)
            code = int(code or 0)
            study.event(f'{verb}_finished', job=args.job, pid=os.getpid(), exit_code=code)
    except StudyLocked as error:
        raise Refused(f'locked: {error}') from None
    except NotImplementedError as error:
        raise Refused(f'not implemented: {error}') from None
    return None, code


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--study', required=True, help='study folder name under studies/')
    p.add_argument('command', choices=['status', 'run', 'verify'])
    p.add_argument('--job', help='run or verify one job of the schedule in this process')
    p.add_argument('--resume', action='store_true', help='rerun stages that a STOP interrupted')
    p.add_argument('--dry-run', action='store_true', help='checks only: no lock, no dispatch')
    q = p.add_argument_group('queue settings (filled by python -m tools.remote launch from remote.toml)')
    q.add_argument('--lanes', help='CUDA device indices, comma-separated')
    q.add_argument('--slots-per-lane', type=int)
    q.add_argument('--memory-cap-gib', type=float)
    q.add_argument('--gpu-free-reserve-gib', type=float)
    q.add_argument('--min-free-disk-gib', type=float)
    return p


def main(argv=None, root=None):
    args = parser().parse_args(argv)
    try:
        study = Study.named(args.study, root=root or ROOT)
        if args.command == 'status':
            print(json.dumps(status(study), indent=2, ensure_ascii=False))
            return 0
        report, code = dispatch(study, args)
    except Refused as refusal:
        print(f'refused: {refusal}', file=sys.stderr)
        return REFUSED
    except StudyStopped as stop:
        print(f'stopped: {stop}', file=sys.stderr)
        return EXIT_STOPPED
    except StudyError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    if report is not None:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
