"""The lane/slot queue with fake workers: capacity, dependencies, stages, failures, STOP, resume, locks and gates."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fakes import ROOT

from studykit.registry import EXIT_STOPPED, Study, StudyError, StudyLocked, StudyStopped, read
from tools.remote.queue import Queue, Settings, settings_from_args

WORKER = r'''
import json, os, sys, time
from pathlib import Path
spec = json.loads(sys.argv[1])
out = Path(spec['out'])
out.mkdir(parents=True, exist_ok=True)
keys = ('CUDA_VISIBLE_DEVICES', 'STUDYKIT_SLOTS', 'STUDYKIT_LANE', 'STUDYKIT_MEMORY_CAP_GIB', 'STUDYKIT_LOCK_HOLDER',
        'STUDYKIT_JOB', 'STUDYKIT_STAGE', 'OMP_NUM_THREADS')
mark = dict(stage=spec['stage'], start=time.time(), pid=os.getpid(), env={k: os.environ.get(k) for k in keys})
stop = Path(spec['stop'])
code = spec.get('exit', 0)
deadline = time.time() + spec.get('seconds', 0.1)
while time.time() < deadline:
    if spec.get('honors_stop', True) and stop.exists():
        code = 75
        break
    time.sleep(0.02)
mark['end'] = time.time()
with open(out / 'marks.jsonl', 'a') as f:
    f.write(json.dumps(mark) + '\n')
sys.exit(code)
'''

HOLDER = r'''
import sys, time
sys.path.insert(0, sys.argv[2])
from tools.remote.agent import try_lock
handle = try_lock(sys.argv[1])
print('locked' if handle else 'busy', flush=True)
time.sleep(60)
'''


def fake_command(study):
    def command(job, stage):
        spec = dict(job.get('behavior', {}).get(stage, {}), stage=stage, out=str(study.run_dir(job['id'])),
                    stop=str(study.stop_path))
        return [sys.executable, '-B', '-c', WORKER, json.dumps(spec)]
    return command


def state_of(study):
    try:
        return read(study.supervisor_path)
    except (OSError, ValueError):  # missing yet, or being replaced
        return {}


def marks(study, job_id):
    path = study.run_dir(job_id) / 'marks.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = Study(self.root / 'studies' / 'queue-study', root=self.root)
        self.study.path.mkdir(parents=True)
        self.holders = []

    def tearDown(self):
        for process in self.holders:
            process.kill()
            process.wait()
        self.tmp.cleanup()

    def settings(self, **overrides):
        values = dict(lanes=(0, 1), slots_per_lane=2, memory_cap_gib=10, poll_seconds=0.02, stop_grace_seconds=0.3,
                      kill_after_seconds=0.5, slot_lock_dir=self.root / 'slots')
        values.update(overrides)
        return Settings(**values)

    def queue(self, jobs, gpu_free=lambda: {0: 100.0, 1: 100.0}, disk_free=lambda: 1000.0, **settings):
        return Queue(self.study, jobs, self.settings(**settings), command=fake_command(self.study),
                     gpu_free=gpu_free, disk_free=disk_free)

    def hold(self, path):
        process = subprocess.Popen([sys.executable, '-B', '-c', HOLDER, str(path), str(ROOT)], stdout=subprocess.PIPE,
                                   text=True)
        self.holders.append(process)
        self.assertEqual(process.stdout.readline().strip(), 'locked')
        return process

    def events(self):
        return [json.loads(line) for line in self.study.events_path.read_text().splitlines()]

    def test_capacity_dependencies_and_stages(self):
        jobs = [dict(id=f'j{i}', behavior=dict(run=dict(seconds=0.3))) for i in range(6)]
        jobs += [dict(id='after', dependencies=['j0', 'j1']), dict(id='no-verify', verify=False),
                 dict(id='wide', slots=2, behavior=dict(run=dict(seconds=0.2)))]
        self.assertEqual(self.queue(jobs).run(), 0)
        state = read(self.study.supervisor_path)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(sorted(state['completed']), sorted(j['id'] for j in jobs))
        intervals = []
        for job in jobs:
            m = {x['stage']: x for x in marks(self.study, job['id'])}
            self.assertEqual(set(m), {'run'} if job['id'] == 'no-verify' else {'run', 'verify'}, job['id'])
            if 'verify' in m:
                self.assertGreaterEqual(m['verify']['start'], m['run']['end'])
            for x in m.values():
                lane, slots = x['env']['CUDA_VISIBLE_DEVICES'], x['env']['STUDYKIT_SLOTS'].split(',')
                intervals += [(x['start'], x['end'], lane, s) for s in slots]
                self.assertEqual(x['env']['STUDYKIT_LOCK_HOLDER'], f'queue-study:{os.getpid()}')
                self.assertEqual(x['env']['OMP_NUM_THREADS'], '4')
            for stage in m:
                record = read(self.study.run_dir(job['id']) / f'stage-{stage}.json')
                self.assertEqual((record['outcome'], record['exit_code']), ('ok', 0))
        wide = marks(self.study, 'wide')[0]['env']
        self.assertEqual((len(wide['STUDYKIT_SLOTS'].split(',')), wide['STUDYKIT_MEMORY_CAP_GIB']), (2, '20'))
        after = min(x['start'] for x in marks(self.study, 'after'))
        self.assertGreaterEqual(after, max(x['end'] for j in ('j0', 'j1') for x in marks(self.study, j)))
        for start, end, lane, slot in intervals:  # never two workers in one slot, never more than four at once
            overlapping = [i for i in intervals if i[0] < end and start < i[1]]
            self.assertLessEqual(len({(i[2], i[3]) for i in overlapping}), 4)
            self.assertEqual(sum(1 for i in overlapping if (i[2], i[3]) == (lane, slot)), 1)
        names = [e['event'] for e in self.events()]
        self.assertEqual((names[0], names[-1]), ('queue_started', 'queue_finished'))
        self.assertEqual(names.count('stage_started'), names.count('stage_finished'))

    def test_failure_blocks_dependents_without_retry(self):
        jobs = [dict(id='bad', behavior=dict(run=dict(exit=1))), dict(id='child', dependencies=['bad']),
                dict(id='other')]
        self.assertEqual(self.queue(jobs).run(), 1)
        state = read(self.study.supervisor_path)
        self.assertEqual(state['status'], 'attention')
        self.assertEqual(state['failed'], {'bad': dict(state='failed', stage='run')})
        self.assertEqual(state['blocked'], ['child'])
        self.assertIn('other', state['completed'])
        self.assertEqual(len(marks(self.study, 'bad')), 1)
        self.assertEqual(marks(self.study, 'child'), [])
        self.assertEqual(self.queue(jobs).run(), 1, 'a failed stage must not be retried')
        self.assertEqual(len(marks(self.study, 'bad')), 1)

    def test_stop_then_resume(self):
        long = dict(run=dict(seconds=30))
        jobs = [dict(id='polite', behavior=long),
                dict(id='stubborn', behavior=dict(run=dict(seconds=30, honors_stop=False)))]
        result = {}
        thread = threading.Thread(target=lambda: result.update(code=self.queue(jobs).run()))
        thread.start()
        deadline = time.time() + 20
        while time.time() < deadline and len(state_of(self.study).get('active', {})) < 2:
            time.sleep(0.05)
        self.study.stop_path.write_text('{"reason": "test"}')
        thread.join(30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result['code'], EXIT_STOPPED)
        state = read(self.study.supervisor_path)
        self.assertEqual((state['status'], sorted(state['stopped'])), ('stopped', ['polite', 'stubborn']))
        self.assertEqual(read(self.study.run_dir('polite') / 'stage-run.json')['exit_code'], EXIT_STOPPED)
        self.assertIn('workers_terminated', [e['event'] for e in self.events()])
        with self.assertRaises(StudyStopped):
            self.queue(jobs).run()
        self.study.stop_path.unlink()
        self.assertEqual(self.queue(jobs).run(), EXIT_STOPPED, 'stopped stages rerun only on resume')
        quick = [dict(j, behavior={}) for j in jobs]
        self.assertEqual(self.queue(quick).run(resume=True), 0)
        self.assertEqual(len(list(self.study.run_dir('polite').glob('stage-run.stopped-*.json'))), 1)

    def test_study_lock_and_frozen_study(self):
        self.hold(self.study.lock_path)
        with self.assertRaises(StudyLocked):
            self.queue([dict(id='a')]).run()
        frozen = Study(self.root / 'studies' / 'brain-sokoban-v1', root=self.root)
        with self.assertRaisesRegex(StudyError, 'frozen'):
            Queue(frozen, [], self.settings()).run()

    def test_slots_held_elsewhere_are_skipped(self):
        self.hold(self.root / 'slots' / 'lane0-slot0.lock')
        jobs = [dict(id=f'j{i}', verify=False) for i in range(3)]
        self.assertEqual(self.queue(jobs, lanes=(0,)).run(), 0)
        self.assertEqual({marks(self.study, j['id'])[0]['env']['STUDYKIT_SLOTS'] for j in jobs}, {'1'})

    def test_gpu_memory_and_disk_gates(self):
        jobs = [dict(id=f'j{i}', verify=False) for i in range(3)]
        self.assertEqual(self.queue(jobs, gpu_free=lambda: {0: 5.0, 1: 100.0}).run(), 0)
        self.assertEqual({marks(self.study, j['id'])[0]['env']['CUDA_VISIBLE_DEVICES'] for j in jobs}, {'1'})
        thread = threading.Thread(target=lambda: self.queue([dict(id='late')], disk_free=lambda: 1.0,
                                                            min_free_disk_gib=10).run())
        thread.start()
        deadline = time.time() + 10
        while time.time() < deadline and state_of(self.study).get('status') != 'waiting_resource':
            time.sleep(0.05)
        self.assertEqual(read(self.study.supervisor_path)['status'], 'waiting_resource')
        self.study.stop_path.write_text('{}')
        thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(marks(self.study, 'late'), [])

    def test_lost_and_live_earlier_workers(self):
        run_dir = self.study.run_dir('a')
        run_dir.mkdir(parents=True)
        (run_dir / 'stage-run.json').write_text(json.dumps(dict(job='a', stage='run', pid=2 ** 31 - 1, outcome=None)))
        jobs = [dict(id='a'), dict(id='b', dependencies=['a'])]
        self.assertEqual(self.queue(jobs).run(), 1)
        state = read(self.study.supervisor_path)
        self.assertEqual(state['failed'], {'a': dict(state='lost', stage='run')})
        self.assertEqual(marks(self.study, 'a'), [])
        live = self.hold(self.root / 'unrelated.lock')
        (run_dir / 'stage-run.json').write_text(json.dumps(dict(job='a', stage='run', pid=live.pid, outcome=None)))
        with self.assertRaisesRegex(StudyError, 'still alive'):
            self.queue(jobs).run()

    def test_bad_jobs_and_settings_from_args(self):
        for jobs in ([dict(id='a', slots=3)], [dict(id='a'), dict(id='a')], [dict(id='a', dependencies=['x'])],
                     [dict(id='a', dependencies=['b']), dict(id='b', dependencies=['a'])], [dict(id='../a')]):
            with self.subTest(jobs=jobs), self.assertRaises(StudyError):
                self.queue(jobs)

        class Args:
            lanes, slots_per_lane, memory_cap_gib = '0,1', 3, 30.0
            gpu_free_reserve_gib, min_free_disk_gib = None, 300.0

        s = settings_from_args(Args, poll_seconds=1.0)
        self.assertEqual((s.lanes, s.slots_per_lane, s.memory_cap_gib, s.gpu_free_reserve_gib, s.min_free_disk_gib,
                          s.poll_seconds), ((0, 1), 3, 30.0, 1.0, 300.0, 1.0))


if __name__ == '__main__':
    unittest.main()
