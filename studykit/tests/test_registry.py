"""Study records: names, protocol, schedule, task lock, amendments, registration, locks and provenance hashing."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from studykit import registry  # noqa: E402
from studykit.registry import (LOCK_ENV, Study, StudyError, StudyLocked, StudyStopped, digest,  # noqa: E402
                               file_lock, lock_state, read, write)

HOLDER = r'''
import sys, time
sys.path.insert(0, sys.argv[2])
from studykit.registry import file_lock
with file_lock(sys.argv[1]):
    print('locked', flush=True)
    time.sleep(60)
'''


class TempStudy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = Study.named('demo-study', root=self.root)
        self.study.path.mkdir(parents=True)
        (self.study.code_dir).mkdir()
        (self.study.code_dir / 'study.py').write_text('X = 1\n')
        self.holders = []

    def tearDown(self):
        for process in self.holders:
            process.kill()
            process.wait()
        self.tmp.cleanup()

    def hold(self, path):
        process = subprocess.Popen([sys.executable, '-B', '-c', HOLDER, str(path), str(ROOT)], stdout=subprocess.PIPE,
                                   text=True)
        self.holders.append(process)
        self.assertEqual(process.stdout.readline().strip(), 'locked')
        return process

    def draft(self, **design):
        self.study.save_protocol(self.study.protocol(**design))
        write(self.study.schedule_path, [dict(id='a'), dict(id='b', dependencies=['a'])])
        self.study.tasks_dir.mkdir(exist_ok=True)
        write(self.study.tasks_dir / 'maps.json', [1, 2, 3])
        self.study.lock_tasks(selection='fixed')


class Records(TempStudy):
    def test_names_and_paths(self):
        for bad in ('Demo', 'a', 'demo_study', '../x', 'demo-', '-demo', 5):
            with self.subTest(name=bad), self.assertRaises(StudyError):
                Study.named(bad, root=self.root)
        self.assertEqual(self.study.relative(self.study.protocol_path), 'studies/demo-study/protocol.json')
        self.assertFalse(self.study.frozen)
        self.assertTrue(Study.named('brain-sokoban-v1', root=self.root).frozen)
        self.assertEqual(Study(ROOT / 'studies' / 'brain-supplement-v2').root, ROOT)
        with self.assertRaises(StudyError):
            self.study.run_dir('../escape')

    def test_atomic_write_and_hashes(self):
        path = self.root / 'x' / 'value.json'
        write(path, dict(b=1, a=[1.5, 'é']))
        self.assertEqual(read(path), dict(b=1, a=[1.5, 'é']))
        self.assertEqual(list(path.parent.iterdir()), [path], 'temporary file left behind')
        with self.assertRaises(ValueError):
            write(path, dict(x=float('nan')))
        self.assertEqual(registry.object_hash(dict(a=1, b=2)), registry.object_hash(dict(b=2, a=1)))

    def test_protocol_hashes_sources_and_stays_frozen(self):
        source = self.root / 'inputs' / 'model.bin'
        source.parent.mkdir()
        source.write_bytes(b'weights')
        value = self.study.protocol(sources=['inputs/model.bin'], seed=10)
        self.assertEqual(value['sources'], {'inputs/model.bin': digest(source)})
        self.assertEqual((value['study'], value['status'], value['seed']), ('demo-study', 'draft', 10))
        with self.assertRaises(ValueError):
            self.study.protocol(sources=['../outside.bin'])
        self.study.save_protocol(value)
        self.assertEqual(self.study.load_protocol()['seed'], 10)
        with self.assertRaises(StudyError):
            self.study.save_protocol(dict(value, status='registered'))
        write(self.study.protocol_path, dict(value, study='other'))
        with self.assertRaisesRegex(StudyError, 'names study'):
            self.study.load_protocol()

    def test_schedule_validation(self):
        bad = ([dict(id='a'), dict(id='a')], [dict(id='a', dependencies=['zz'])],
               [dict(id='a', dependencies=['b']), dict(id='b', dependencies=['a'])], [dict(id='x y')], [dict()], {})
        for jobs in bad:
            with self.subTest(jobs=jobs), self.assertRaises(StudyError):
                write(self.study.schedule_path, jobs)
                self.study.schedule()

    def test_task_lock(self):
        self.draft()
        self.assertTrue(self.study.check_task_lock()['ok'])
        write(self.study.tasks_dir / 'maps.json', [1, 2, 4])
        write(self.study.tasks_dir / 'extra.json', [])
        result = self.study.check_task_lock()
        self.assertEqual((result['ok'], result['changed'], result['unlocked']), (False, ['maps.json'], ['extra.json']))

    def test_registration_and_amendments(self):
        self.draft(seed=1)
        record = self.study.register(note='test')
        self.assertEqual(self.study.load_protocol()['status'], 'registered')
        self.assertEqual(record['protocol_sha256'], digest(self.study.protocol_path))
        self.assertTrue(self.study.check_registration()['ok'])
        with self.assertRaises(StudyError):
            self.study.register()
        first = self.study.amend('Fix n after the pilot', dict(n_primary=10))
        second = self.study.amend('Second change', dict(x=1))
        chain = self.study.amendments()
        self.assertEqual([a['number'] for a in chain], [1, 2])
        self.assertEqual(chain[1]['previous_sha256'], digest(first))
        self.assertEqual(self.study.check_registration()['amendments_since'], 2)
        events = [json.loads(line)['event'] for line in self.study.events_path.read_text().splitlines()]
        self.assertEqual(events, ['registered', 'amended', 'amended'])
        value = read(first)
        value['changes'] = dict(n_primary=12)
        write(first, value)
        with self.assertRaisesRegex(StudyError, 'chain is broken'):
            self.study.amendments()
        write(first, dict(value, changes=dict(n_primary=10)))  # restoring the exact bytes is the only way back
        self.assertEqual([a['file'] for a in self.study.amendments()], [first.name, second.name])
        write(self.study.schedule_path, [dict(id='a')])
        self.assertEqual(self.study.check_registration()['changed'], ['schedule_sha256'])

    def test_stop(self):
        self.assertFalse(self.study.stopped())
        self.study.stop_path.write_text('{}')
        with self.assertRaises(StudyStopped):
            self.study.check_stop()


class Locks(TempStudy):
    def test_worker_lock_is_reentrant_and_exclusive(self):
        self.assertEqual(self.study.lock_state(), 'absent')
        with self.study.worker_lock():
            self.assertEqual(os.environ[LOCK_ENV], f'demo-study:{os.getpid()}')
            with self.study.worker_lock():
                self.assertEqual(self.study.lock_state(), 'held_here')
            self.assertEqual(self.study.lock_state(), 'held_here')
            child = subprocess.run([sys.executable, '-B', '-c',
                                    'import os,sys; sys.path.insert(0, sys.argv[1]); from studykit.registry import Study; '
                                    'print(Study.named("demo-study", root=sys.argv[2]).held_by_parent())',
                                    str(ROOT), str(self.root)], capture_output=True, text=True)
            self.assertEqual(child.stdout.strip(), 'True', child.stderr)
        self.assertNotIn(LOCK_ENV, os.environ)
        self.assertEqual(self.study.lock_state(), 'free')
        self.assertFalse(self.study.held_by_parent())

    def test_other_process_holds_the_lock(self):
        self.hold(self.study.lock_path)
        self.assertEqual(lock_state(self.study.lock_path), 'held')
        with self.assertRaises(StudyLocked):
            with self.study.worker_lock():
                pass
        with self.assertRaises(StudyLocked):
            with file_lock(self.study.lock_path):
                pass


class Provenance(TempStudy):
    def module(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
        spec = importlib.util.spec_from_file_location(f'tmpmod_{abs(hash(str(path)))}', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_loaded_project_modules_are_hashed_and_snapshotted(self):
        loaded = self.module('sokoban/helper.py', 'Y = 2\n')
        venv = self.module('.venv-linux/lib/site-packages/dep.py', 'Z = 3\n')
        outside = Path(registry.__file__)
        key = self.study.provenance(modules=[loaded, venv, outside])
        manifest = self.study.snapshot(key)
        self.assertEqual(sorted(manifest), ['sokoban/helper.py', 'studies/demo-study/code/study.py'])
        self.assertEqual(key, self.study.provenance(modules=[loaded, venv, outside]), 'key not deterministic')
        with zipfile.ZipFile(self.study.snapshots_dir / key / 'source.zip') as z:
            self.assertEqual(sorted(z.namelist()), sorted(manifest))
            self.assertEqual(z.read('sokoban/helper.py'), b'Y = 2\n')
        zip_bytes = (self.study.snapshots_dir / key / 'source.zip').read_bytes()
        self.assertEqual(registry._zip_bytes(self.root, [self.root / p for p in manifest]), zip_bytes)
        self.assertEqual(self.study.changed_files(key), [])
        (self.root / 'sokoban' / 'helper.py').write_bytes(b'Y = 3\n')
        self.assertEqual(self.study.changed_files(key), [dict(path='sokoban/helper.py', reason='changed')])
        other = self.study.provenance(modules=[loaded])
        self.assertNotEqual(other, key)
        self.assertEqual(len([p for p in self.study.snapshots_dir.iterdir() if p.is_dir()]), 2)
        self.assertFalse(list(self.study.snapshots_dir.glob('.*.partial')))

    def test_sys_modules_by_default(self):
        loaded = self.module('lib/thing.py', 'T = 1\n')
        sys.modules['tmp_registry_thing'] = loaded
        try:
            manifest = self.study.snapshot(self.study.provenance())
        finally:
            del sys.modules['tmp_registry_thing']
        self.assertIn('lib/thing.py', manifest)
        self.assertNotIn('studykit/registry.py', manifest, 'files outside the study root were hashed')

    def test_real_project_modules(self):
        files = registry.project_module_files(ROOT)
        rels = {p.relative_to(ROOT).as_posix() for p in files}
        self.assertIn('studykit/registry.py', rels)
        self.assertFalse(any('site-packages' in r or r.startswith('MIGRATION/python') for r in rels))


if __name__ == '__main__':
    unittest.main()
