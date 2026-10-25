"""research.py: read-only status, refusals, dispatch to the study's code, and the per-study worker lock."""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from studykit.registry import EXIT_STOPPED, LOCK_ENV, Study, read, write  # noqa: E402

spec = importlib.util.spec_from_file_location('research_under_test', ROOT / 'research.py')
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)

STUDY_CODE = r'''
import json, os
from pathlib import Path

def _note(study, name, **extra):
    with open(study.path / 'calls.jsonl', 'a') as f:
        f.write(json.dumps(dict(call=name, lock_env=os.environ.get('STUDYKIT_LOCK_HOLDER'), **extra)) + '\n')

def run(study, args):
    _note(study, 'run', lock=study.lock_state())
    return 0

def run_job(study, job, args):
    _note(study, 'run_job', job=job['id'], lock=study.lock_state(), parent=study.held_by_parent())
    if job['id'] == 'stops':
        from studykit.registry import StudyStopped
        raise StudyStopped('STOP seen at a safe point')
    if job['id'] == 'todo':
        raise NotImplementedError('not yet')
    return 0

def verify(study, args):
    _note(study, 'verify')
    return 3
'''

HOLDER = r'''
import sys, time
sys.path.insert(0, sys.argv[2])
from studykit.registry import file_lock
with file_lock(sys.argv[1]):
    print('locked', flush=True)
    time.sleep(60)
'''


def call(argv, root):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = research.main(argv, root=root)
    return code, out.getvalue(), err.getvalue()


def listing(path):
    return sorted((p.relative_to(path).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in Path(path).rglob('*'))


class Scaffolds(unittest.TestCase):
    def test_status_of_the_new_studies_is_a_read_only_dry_run(self):
        for name in ('brain-supplement-v2', 'brain-pathology-v1'):
            folder = ROOT / 'studies' / name
            before = listing(folder)
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(ROOT / 'research.py'), '--study', name,
                                     'status'], capture_output=True, text=True, cwd=str(ROOT))
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads(result.stdout)
            self.assertEqual((status['study'], status['dry_run'], status['frozen']), (name, True, False))
            self.assertEqual(status['protocol']['status'], 'draft')
            self.assertEqual(status['schedule']['jobs'], 0)
            self.assertTrue(status['task_lock']['ok'])
            self.assertGreaterEqual(status['test_plan']['families'], 0)
            self.assertEqual(status['queue']['worker_lock'], 'absent')
            self.assertEqual(listing(folder), before, f'{name}: status changed files')

    def test_scaffold_records(self):
        for name in ('brain-supplement-v2', 'brain-pathology-v1'):
            study = Study.named(name)
            protocol = study.load_protocol()
            self.assertEqual((protocol['study'], protocol['status']), (name, 'draft'))
            self.assertEqual(study.schedule(), [])
            self.assertTrue(study.check_task_lock()['ok'])        # every task file locked and unchanged
            self.assertTrue(study.amendments_dir.is_dir())
            # plan v2 amendments: amendments() checks the numbering and the hash chain; each record is this study's
            self.assertEqual({a['study'] for a in study.amendments()} - {name}, set())
            self.assertTrue((study.path / 'README.md').is_file())
            for absent in ('registration.json', 'STOP', 'worker.lock', 'events.jsonl'):
                self.assertFalse((study.path / absent).exists(), f'{name}/{absent}')
            # before registration, runs/ holds only the named campaigns of plan v2 (bring-up checks, credibility tests,
            # Phase 3 and Phase 4 runs, X1-X10 replicates, controls and difficulty sets, regime-B/C and rescue training,
            # O14, T-DY1/T-DY2 and the single-cell runs), never an unnamed or queue-registered job
            runs = study.path / 'runs'
            if runs.exists():
                known = ('g0-', 'g4-', 'phase2-', 'phase3-', 'phase4-', 'x1-', 'x2-', 'x3-', 'x4-', 'x5-', 'x6-', 'x8-',
                         'x9-', 'b-', 'regime-b-', 'regime-c', 'rescue', 'controls-', 'difficulty-', 'o14-', 'dy1-',
                         'dy2-', 'sc-')
                self.assertEqual([p.name for p in runs.iterdir() if not p.name.startswith(known)], [], f'{name}/runs')
        code, _, err = call(['--study', 'brain-supplement-v2', 'run', '--dry-run'], ROOT)
        self.assertEqual(code, research.REFUSED)
        self.assertIn('draft', err)

    def test_frozen_study_is_read_only(self):
        folder = ROOT / 'studies' / 'brain-sokoban-v1'
        if not (folder / 'STOP').exists():
            self.skipTest('the frozen study STOP file is local only (never pushed to the GPU host)')
        top = sorted(p.name for p in folder.iterdir())
        stop = (folder / 'STOP').read_bytes()
        code, out, _ = call(['--study', 'brain-sokoban-v1', 'status'], ROOT)
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out)['frozen'])
        for command in ('run', 'verify'):
            code, _, err = call(['--study', 'brain-sokoban-v1', command], ROOT)
            self.assertEqual(code, research.REFUSED)
            self.assertIn('frozen', err)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), top)
        self.assertEqual((folder / 'STOP').read_bytes(), stop)


class Dispatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = Study.named('demo-study', root=self.root)
        self.study.code_dir.mkdir(parents=True)
        (self.study.code_dir / 'study.py').write_text(STUDY_CODE)
        self.study.save_protocol(self.study.protocol(seed=1))
        write(self.study.schedule_path, [dict(id='job-1'), dict(id='stops'), dict(id='todo')])
        self.study.lock_tasks()
        self.holders = []

    def tearDown(self):
        for process in self.holders:
            process.kill()
            process.wait()
        self.tmp.cleanup()

    def calls(self):
        path = self.study.path / 'calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_draft_is_refused_then_registered_study_runs(self):
        code, _, err = call(['--study', 'demo-study', 'run'], self.root)
        self.assertEqual(code, research.REFUSED)
        self.assertIn('draft', err)
        self.assertFalse(self.study.lock_path.exists(), 'a refusal must not take the lock')
        self.study.register()
        code, out, _ = call(['--study', 'demo-study', 'run', '--dry-run'], self.root)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)['would_call'], 'studies/demo-study/code/study.py:run')
        self.assertEqual(self.calls(), [])
        code, _, _ = call(['--study', 'demo-study', 'run'], self.root)
        self.assertEqual(code, 0)
        self.assertEqual(self.calls()[-1], dict(call='run', lock='held_here', lock_env=f'demo-study:{os.getpid()}'))
        self.assertNotIn(LOCK_ENV, os.environ)
        code, _, _ = call(['--study', 'demo-study', 'verify'], self.root)
        self.assertEqual(code, 3, 'the study code\'s exit code is returned')
        events = [json.loads(line)['event'] for line in self.study.events_path.read_text().splitlines()]
        self.assertEqual(events, ['registered', 'run_started', 'run_finished', 'verify_started', 'verify_finished'])

    def test_jobs_take_the_study_lock_unless_their_parent_holds_it(self):
        self.study.register()
        code, _, _ = call(['--study', 'demo-study', 'run', '--job', 'job-1'], self.root)
        self.assertEqual(code, 0)
        self.assertEqual(self.calls()[-1]['lock'], 'held_here')
        self.assertFalse(self.calls()[-1]['parent'])
        self.assertTrue((self.study.run_dir('job-1') / 'job.lock').exists())
        with self.study.worker_lock():  # as a queue supervisor would: its child must not need the lock
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c',
                                     'import sys, importlib.util; spec = importlib.util.spec_from_file_location('
                                     '"r", sys.argv[1]); r = importlib.util.module_from_spec(spec); '
                                     'spec.loader.exec_module(r); sys.exit(r.main(sys.argv[2:-1], root=sys.argv[-1]))',
                                     str(ROOT / 'research.py'), '--study', 'demo-study', 'run', '--job', 'job-1',
                                     str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.calls()[-1]['parent'])
        code, _, err = call(['--study', 'demo-study', 'run', '--job', 'nope'], self.root)
        self.assertEqual(code, research.REFUSED)
        code, _, err = call(['--study', 'demo-study', 'run', '--job', 'stops'], self.root)
        self.assertEqual(code, EXIT_STOPPED)
        code, _, err = call(['--study', 'demo-study', 'run', '--job', 'todo'], self.root)
        self.assertEqual(code, research.REFUSED)
        self.assertIn('not implemented', err)

    def test_lock_stop_and_changed_records_refuse(self):
        self.study.register()
        process = subprocess.Popen([sys.executable, '-B', '-c', HOLDER, str(self.study.lock_path), str(ROOT)],
                                   stdout=subprocess.PIPE, text=True)
        self.holders.append(process)
        self.assertEqual(process.stdout.readline().strip(), 'locked')
        code, _, err = call(['--study', 'demo-study', 'run'], self.root)
        self.assertEqual(code, research.REFUSED)
        self.assertIn('locked', err)
        process.kill()
        process.wait()
        self.study.stop_path.write_text('{}')
        code, _, err = call(['--study', 'demo-study', 'verify'], self.root)
        self.assertEqual(code, research.REFUSED)
        self.assertIn('STOP', err)
        self.study.stop_path.unlink()
        write(self.study.schedule_path, [dict(id='job-1')])
        code, _, err = call(['--study', 'demo-study', 'run'], self.root)
        self.assertEqual(code, research.REFUSED)
        self.assertIn('schedule_sha256', err)
        code, _, err = call(['--study', 'not-a-study', 'run'], self.root)
        self.assertEqual(code, research.REFUSED)


if __name__ == '__main__':
    unittest.main()
