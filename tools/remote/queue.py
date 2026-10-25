"""Lane/slot supervisor of one registered study on the GPU host.

Each GPU lane (a CUDA device index) offers ``slots_per_lane`` worker positions. A job takes one slot, or
``"slots": k`` slots on one lane, and runs its stages as fresh processes: 'run', then an independent 'verify'
unless the job sets ``"verify": false``. Jobs start in schedule order once their ``dependencies`` are complete.

Locks: the study's worker lock (studies/<name>/worker.lock, one supervisor per study) and one lock per slot
(runtime/gpu-slots/lane<L>-slot<S>.lock, so no slot is shared between studies). On POSIX each worker inherits both
lock descriptors, so they stay held while it lives even if the supervisor dies, and a new supervisor refuses to
start while workers of an earlier one are alive. A failed stage is never retried; its dependents are blocked.

Records: studies/<name>/supervisor.json (rewritten every tick), studies/<name>/events.jsonl (append-only),
runs/<job>/stage-<stage>.json (written at the start and the end of each stage) and runs/<job>/<stage>.log.

STOP (studies/<name>/STOP): no stage starts; running workers see the same file and exit at a safe point with
EXIT_STOPPED; after ``stop_grace_seconds`` the supervisor terminates what is left. Stopped stages run again only
with ``resume=True``. ``Queue.run`` returns 0 when every job is complete, 1 when something needs attention (a failed
or lost stage, or jobs blocked by one) and EXIT_STOPPED when it stopped.
"""
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from studykit.registry import EXIT_STOPPED, StudyError, check_jobs, read, write

from .agent import pid_alive, release, try_lock

STAGES = ('run', 'verify')


@dataclass
class Settings:
    lanes: tuple = (0,)
    slots_per_lane: int = 1
    memory_cap_gib: float = None      # per slot; a job's workers get cap x slots as STUDYKIT_MEMORY_CAP_GIB
    gpu_free_reserve_gib: float = 1.0
    min_free_disk_gib: float = 0.0
    poll_seconds: float = 2.0
    stop_grace_seconds: float = 120.0
    kill_after_seconds: float = 10.0
    threads_per_worker: int = 4
    slot_lock_dir: Path = None        # default <project root>/runtime/gpu-slots

    @classmethod
    def from_config(cls, config, **overrides):
        """Settings from a tools.remote.config.RemoteConfig."""
        values = dict(lanes=tuple(config.lanes), slots_per_lane=config.slots_per_lane,
                      memory_cap_gib=config.memory_cap_gib, gpu_free_reserve_gib=config.gpu_free_reserve_gib,
                      min_free_disk_gib=config.min_free_disk_gib)
        values.update(overrides)
        return cls(**values)


def settings_from_args(args, **overrides):
    """Settings from research.py's queue options (tools.remote launch fills them from remote.toml)."""
    values = {}
    if getattr(args, 'lanes', None):
        values['lanes'] = tuple(int(x) for x in str(args.lanes).split(','))
    for name in ('slots_per_lane', 'memory_cap_gib', 'gpu_free_reserve_gib', 'min_free_disk_gib'):
        if getattr(args, name, None) is not None:
            values[name] = getattr(args, name)
    values.update(overrides)
    return Settings(**values)


def nvidia_free_gib():
    """{device index: free GiB} from nvidia-smi, or None when it cannot be queried."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return {int(i): float(m) / 1024 for i, m in (line.split(',') for line in out.strip().splitlines())}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def default_judge(job, stage, exit_code, out):
    """Stage outcome from the exit code: 0 ok, EXIT_STOPPED stopped, anything else failed."""
    return 'ok' if exit_code == 0 else 'stopped' if exit_code == EXIT_STOPPED else 'failed'


def status(study):
    """Read-only state of a study's queue (no lock is taken and nothing is written)."""
    supervisor = read(study.supervisor_path) if study.supervisor_path.exists() else None
    pid = (supervisor or {}).get('pid')
    return dict(supervisor=supervisor, supervisor_alive=pid_alive(pid) if isinstance(pid, int) else False,
                stop=study.stopped(), worker_lock=study.lock_state())


class Queue:
    """Runs the jobs of one study on the configured lanes and slots."""

    def __init__(self, study, jobs, settings=None, command=None, judge=None, gpu_free=nvidia_free_gib,
                 disk_free=None, env=None):
        self.study = study
        self.jobs = check_jobs(list(jobs))
        self.by_id = {j['id']: j for j in self.jobs}
        self.settings = settings or Settings()
        s = self.settings
        if not s.lanes or s.slots_per_lane < 1:
            raise StudyError('the queue needs at least one lane and one slot per lane')
        for job in self.jobs:
            slots = job.get('slots', 1)
            if not isinstance(slots, int) or isinstance(slots, bool) or not 1 <= slots <= s.slots_per_lane:
                raise StudyError(f"job {job['id']}: slots must be an integer from 1 to {s.slots_per_lane}")
        self.command = command or self.default_command
        self.judge = judge or default_judge
        self.gpu_free = gpu_free
        self.disk_free = disk_free or (lambda: shutil.disk_usage(self.study.path).free / 2 ** 30)
        self.env = dict(env or {})
        self.slot_dir = Path(s.slot_lock_dir) if s.slot_lock_dir else study.root / 'runtime' / 'gpu-slots'
        self.active = {}  # job id -> running stage
        self.notes = set()

    # ------------------------------------------------------------ job state (from the stage records)

    def default_command(self, job, stage):
        return [sys.executable, '-B', '-X', 'utf8', '-u', str(self.study.root / 'research.py'), '--study',
                self.study.name, stage, '--job', job['id']]

    def stages(self, job):
        return STAGES if job.get('verify', True) else STAGES[:1]

    def record_path(self, job_id, stage):
        return self.study.run_dir(job_id) / f'stage-{stage}.json'

    def state(self, job):
        """('complete', None) or (state, stage) with state pending, running, lost, failed or stopped."""
        for stage in self.stages(job):
            path = self.record_path(job['id'], stage)
            if not path.exists():
                return 'pending', stage
            record = read(path)
            outcome = record.get('outcome')
            if outcome == 'ok':
                continue
            if outcome is None:
                alive = job['id'] in self.active or pid_alive(record.get('pid'))
                return ('running' if alive else 'lost'), stage
            return outcome, stage
        return 'complete', None

    def ready(self, job, states):
        state, _ = states[job['id']]
        return state == 'pending' and all(states[d][0] == 'complete' for d in job.get('dependencies', []))

    # ------------------------------------------------------------ processes

    def _place(self, need, free):
        """(lane, slots, lock handles) with ``need`` free slots on one lane, or None."""
        s = self.settings
        for lane in s.lanes:
            if free is not None and s.memory_cap_gib and lane in free:
                if free[lane] < need * s.memory_cap_gib + s.gpu_free_reserve_gib:
                    continue
            ours = {slot for a in self.active.values() if a['lane'] == lane for slot in a['slots']}
            chosen, handles = [], []
            for slot in range(s.slots_per_lane):
                if slot in ours:
                    continue
                handle = try_lock(self.slot_dir / f'lane{lane}-slot{slot}.lock')
                if handle is None:
                    continue  # held by a worker of another study or of an earlier supervisor
                chosen.append(slot)
                handles.append(handle)
                if len(chosen) == need:
                    return lane, chosen, handles
            for handle in handles:
                release(handle)
        return None

    def _start(self, job, stage, lane, slots, handles, study_lock):
        s = self.settings
        out = self.study.run_dir(job['id'])
        out.mkdir(parents=True, exist_ok=True)
        argv = [str(a) for a in self.command(job, stage)]
        threads = str(s.threads_per_worker)
        env = dict(os.environ)
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(lane), STUDYKIT_STUDY=self.study.name,
                   STUDYKIT_JOB=job['id'], STUDYKIT_STAGE=stage, STUDYKIT_LANE=str(lane),
                   STUDYKIT_SLOTS=','.join(map(str, slots)), PYTHONUNBUFFERED='1', OMP_NUM_THREADS=threads,
                   MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
        env.update(self.env)
        if s.memory_cap_gib:
            env['STUDYKIT_MEMORY_CAP_GIB'] = f'{s.memory_cap_gib * len(slots):g}'
        log = (out / f'{stage}.log').open('ab')
        kwargs = dict(cwd=str(self.study.root), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if os.name == 'posix':
            kwargs.update(start_new_session=True, pass_fds=[study_lock.fileno()] + [h.fileno() for h in handles])
        else:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            process = subprocess.Popen(argv, **kwargs)
        except OSError:
            log.close()
            for handle in handles:
                release(handle)
            raise
        record = dict(job=job['id'], stage=stage, lane=lane, slots=slots, pid=process.pid, command=argv,
                      host=platform.node(), started_unix=time.time(), finished_unix=None, exit_code=None,
                      outcome=None)
        write(self.record_path(job['id'], stage), record)
        self.active[job['id']] = dict(stage=stage, process=process, lane=lane, slots=slots, handles=handles, log=log,
                                      record=record, terminated=False)
        self.study.event('stage_started', job=job['id'], stage=stage, lane=lane, slots=slots, pid=process.pid)

    def _reap(self):
        for job_id, a in list(self.active.items()):
            code = a['process'].poll()
            if code is None:
                continue
            out = self.study.run_dir(job_id)
            outcome = 'stopped' if a['terminated'] else self.judge(self.by_id[job_id], a['stage'], code, out)
            record = dict(a['record'], finished_unix=time.time(), exit_code=code, outcome=outcome)
            write(self.record_path(job_id, a['stage']), record)
            for handle in a['handles']:
                release(handle)
            a['log'].close()
            del self.active[job_id]
            self.study.event('stage_finished', job=job_id, stage=a['stage'], exit_code=code, outcome=outcome,
                             seconds=round(record['finished_unix'] - record['started_unix'], 3))

    def _signal(self, kill):
        for a in self.active.values():
            process = a['process']
            if process.poll() is not None:
                continue
            a['terminated'] = True
            try:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
                elif kill:
                    process.kill()
                else:
                    process.terminate()
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def _launch(self, states, study_lock):
        """Start every ready stage that fits; returns why something ready could not start (None if nothing waits)."""
        s = self.settings
        if self.disk_free() < s.min_free_disk_gib:
            return 'disk'
        free = self.gpu_free() if self.gpu_free else None
        if free is None and self.gpu_free and 'gpu_query' not in self.notes:
            self.notes.add('gpu_query')
            self.study.event('gpu_query_unavailable', note='free-memory gate skipped')
        free = dict(free) if free else None
        waiting = None
        for job in self.jobs:
            if job['id'] in self.active or not self.ready(job, states):
                continue
            need = job.get('slots', 1)
            placed = self._place(need, free)
            if placed is None:
                waiting = 'gpu'
                continue
            lane, slots, handles = placed
            self._start(job, states[job['id']][1], lane, slots, handles, study_lock)
            states[job['id']] = ('running', states[job['id']][1])
            if free is not None and s.memory_cap_gib and lane in free:
                free[lane] -= need * s.memory_cap_gib  # nvidia-smi does not see the new worker yet
        return waiting

    # ------------------------------------------------------------ supervisor loop

    def snapshot(self, states, status, stopping, resume):
        s = self.settings
        failed = {k: dict(state=v[0], stage=v[1]) for k, v in states.items() if v[0] in ('failed', 'lost')}
        complete = {k for k, v in states.items() if v[0] == 'complete'}
        blocked = [j['id'] for j in self.jobs if states[j['id']][0] == 'pending' and not self.ready(j, states)
                   and any(states[d][0] in ('failed', 'lost', 'stopped') for d in j.get('dependencies', []))]
        return dict(status=status, study=self.study.name, pid=os.getpid(), host=platform.node(),
                    started_unix=self.started, updated_unix=time.time(), resume=resume, lanes=list(s.lanes),
                    slots_per_lane=s.slots_per_lane, memory_cap_gib=s.memory_cap_gib, stop_requested=stopping,
                    completed=sorted(complete), failed=failed,
                    stopped=sorted(k for k, v in states.items() if v[0] == 'stopped'),
                    ready=[j['id'] for j in self.jobs if j['id'] not in self.active and self.ready(j, states)],
                    blocked=blocked,
                    pending=sorted(k for k, v in states.items() if v[0] == 'pending'),
                    active={k: dict(job=k, stage=a['stage'], lane=a['lane'], slots=a['slots'], pid=a['process'].pid,
                                    started_unix=a['record']['started_unix']) for k, a in self.active.items()})

    def _status(self, states, stopping, waiting):
        if self.active:
            return 'stopping' if stopping else 'running'
        if all(v[0] == 'complete' for v in states.values()):
            return 'complete'
        if stopping:
            return 'stopped'
        if any(self.ready(j, states) for j in self.jobs):
            return 'waiting_resource' if waiting else 'running'
        if any(v[0] in ('failed', 'lost') for v in states.values()):
            return 'attention'
        if any(v[0] == 'stopped' for v in states.values()):
            return 'stopped'
        return 'attention'

    def run(self, resume=False):
        """Supervise until every job is complete, nothing more can run, or STOP has emptied the slots."""
        study, s = self.study, self.settings
        if study.frozen:
            raise StudyError(f'{study.name} is frozen; its queue never runs again')
        with study.worker_lock() as study_lock:
            study.check_stop()
            states = {j['id']: self.state(j) for j in self.jobs}
            alive = sorted(k for k, v in states.items() if v[0] == 'running')
            if alive:
                raise StudyError(f'workers of an earlier supervisor are still alive ({alive}); never launch twice')
            if resume:
                stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
                for job_id, (state, stage) in states.items():
                    if state == 'stopped':
                        path = self.record_path(job_id, stage)
                        os.replace(path, path.with_name(f'stage-{stage}.stopped-{stamp}.json'))
                        states[job_id] = ('pending', stage)
            self.started = time.time()
            study.event('queue_started', pid=os.getpid(), host=platform.node(), lanes=list(s.lanes),
                        slots_per_lane=s.slots_per_lane, memory_cap_gib=s.memory_cap_gib, jobs=len(self.jobs),
                        resume=resume)
            stop_seen = terminated_at = None
            killed = False
            try:
                while True:
                    self._reap()
                    states = {j['id']: self.state(j) for j in self.jobs}
                    stopping = study.stopped()
                    now = time.time()
                    if stopping and stop_seen is None:
                        stop_seen = now
                        study.event('stop_observed', active=sorted(self.active))
                    if stopping and self.active and terminated_at is None and now - stop_seen >= s.stop_grace_seconds:
                        terminated_at = now
                        study.event('workers_terminated', active=sorted(self.active))
                        self._signal(kill=False)
                    if terminated_at is not None and self.active and not killed and \
                            now - terminated_at >= s.kill_after_seconds:
                        killed = True
                        self._signal(kill=True)
                    waiting = None if stopping else self._launch(states, study_lock)
                    status = self._status(states, stopping, waiting)
                    write(study.supervisor_path, self.snapshot(states, status, stopping, resume))
                    if not self.active and status in ('complete', 'stopped', 'attention'):
                        study.event('queue_finished', status=status,
                                    completed=sum(v[0] == 'complete' for v in states.values()), jobs=len(self.jobs))
                        return {'complete': 0, 'attention': 1, 'stopped': EXIT_STOPPED}[status]
                    time.sleep(s.poll_seconds)
            except BaseException as error:
                # Workers keep running and keep their locks; a new supervisor refuses to start until they end.
                write(study.supervisor_path, dict(self.snapshot(states, 'supervisor_exited', study.stopped(), resume),
                                                  error=repr(error)))
                study.event('supervisor_exited', error=repr(error), active=sorted(self.active))
                raise
            finally:
                for a in self.active.values():
                    a['log'].close()
