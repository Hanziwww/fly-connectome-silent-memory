"""verify_sync keeps the schema of tools/verify_remote_sync.py; the status, stop, launch and ping operations."""
import json
import sys
import unittest

from fakes import Workspace, make_config, write

from tools.remote import RemoteError, sync, verify_sync
from tools.remote.ssh import SSH

# Keys and types of sync-verification.json as written by tools/verify_remote_sync.py.
OLD_SCHEMA = dict(transport_verified=bool, scientific_acceptance_separate=bool, files=int, bytes=int, missing=list,
                  mismatches=list, manifest=dict)


class VerifySync(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.ssh = SSH(make_config(), self.ws.executor)
        self.root = self.ws.project
        self.study = self.ws.remote / 'studies' / 'demo-study'
        write(self.study / 'runs' / 'job-1' / 'result.json', '{"status": "ok"}\n')
        write(self.study / 'runs' / 'job-1' / 'model.pt', b'\x00\x01' * 50000)
        write(self.study / 'events.jsonl', '{"event": "queue_finished"}\n')
        write(self.study / 'supervisor.json', json.dumps(dict(status='complete', active={}, pid=0)))

    def tearDown(self):
        self.ws.close()

    def test_schema_and_detection(self):
        sync.pull(self.ssh, 'demo-study', root=self.root)
        result = verify_sync.verify(self.ssh, 'demo-study', root=self.root)
        self.assertEqual(list(result), list(OLD_SCHEMA))
        for key, kind in OLD_SCHEMA.items():
            self.assertIsInstance(result[key], kind, key)
        self.assertTrue(result['transport_verified'])
        self.assertEqual(result['files'], 4)
        self.assertIn('studies/demo-study/runs/job-1/model.pt', result['manifest'])
        saved = json.loads((self.root / 'studies/demo-study/sync-verification.json').read_text())
        self.assertEqual(saved, result)
        self.assertEqual(verify_sync.verified_manifest('demo-study', self.root,
                                                       require=['studies/demo-study/runs/job-1/model.pt']),
                         result['manifest'])
        local = self.root / 'studies' / 'demo-study' / 'runs' / 'job-1'
        write(local / 'model.pt', b'tampered')
        (local / 'result.json').unlink()
        result = verify_sync.verify(self.ssh, 'demo-study', root=self.root)
        self.assertFalse(result['transport_verified'])
        self.assertEqual(result['missing'], ['studies/demo-study/runs/job-1/result.json'])
        self.assertEqual(result['mismatches'], ['studies/demo-study/runs/job-1/model.pt'])
        with self.assertRaisesRegex(RemoteError, 'failed'):
            verify_sync.verified_manifest('demo-study', self.root)

    def test_requires_an_ended_queue(self):
        write(self.study / 'supervisor.json', json.dumps(dict(status='running', active={'job-2': dict(job='job-2')})))
        with self.assertRaisesRegex(RemoteError, 'has not ended'):
            verify_sync.verify(self.ssh, 'demo-study', root=self.root)
        (self.study / 'supervisor.json').unlink()
        with self.assertRaisesRegex(RemoteError, 'nothing has run'):
            verify_sync.verify(self.ssh, 'demo-study', root=self.root)
        with self.assertRaises(RemoteError):
            verify_sync.verified_manifest('demo-study', self.root)

    def test_compare_matches_the_old_definition(self):
        manifest = {'studies/demo-study/a.json': dict(bytes=2, sha256='0' * 64)}
        result = verify_sync.compare(manifest, self.root)
        self.assertEqual(result['missing'], ['studies/demo-study/a.json'])
        self.assertEqual((result['files'], result['bytes']), (1, 2))


class RemoteOperations(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.ssh = SSH(make_config(), self.ws.executor)
        self.study = self.ws.remote / 'studies' / 'demo-study'
        write(self.study / 'protocol.json', '{}')

    def tearDown(self):
        self.ws.close()

    def test_ping(self):
        result = self.ssh.call('ping', python='/srv/flybrain/.venv-linux/bin/python')
        self.assertTrue(result['ok'])
        self.assertTrue(result['research_python_exists'])
        self.assertIsNone(result['sync'])

    def test_stop_status_and_launch(self):
        write(self.ws.remote / 'research.py', 'import sys, json\n'
              'open("launched.json", "w").write(json.dumps(sys.argv))\n')
        stop = self.ssh.call('stop', study='demo-study', reason='pause for test')
        self.assertTrue(stop['created'])
        self.assertFalse(self.ssh.call('stop', study='demo-study')['created'])
        record = json.loads((self.study / 'STOP').read_text())
        self.assertEqual(record['reason'], 'pause for test')
        self.assertTrue(record['resume_requires_explicit_user_request'])
        state = self.ssh.call('status', study='demo-study')
        self.assertTrue(state['stop'])
        self.assertEqual(state['worker_lock'], 'free')
        with self.assertRaisesRegex(RemoteError, 'STOP'):
            self.ssh.call('launch', study='demo-study', python='/srv/flybrain/.venv-linux/bin/python', args=[])
        launched = self.ssh.call('launch', study='demo-study', python='/srv/flybrain/.venv-linux/bin/python',
                                 args=['--lanes', '0', '--slots-per-lane', '1'], clear_stop=True, settle_seconds=3)
        self.assertEqual(launched['exit_code'], 0, launched)
        self.assertFalse((self.study / 'STOP').exists())
        self.assertTrue(launched['stop_cleared'])
        argv = json.loads((self.ws.remote / 'launched.json').read_text())
        self.assertEqual(argv[1:], ['--study', 'demo-study', 'run', '--lanes', '0', '--slots-per-lane', '1'])
        with self.assertRaisesRegex(RemoteError, 'invalid queue arguments'):
            self.ssh.call('launch', study='demo-study', python='/srv/flybrain/.venv-linux/bin/python',
                          args=['--lanes', '0; rm -rf /'])
        for op in ('stop', 'launch'):
            with self.subTest(op=op), self.assertRaisesRegex(RemoteError, 'frozen'):
                self.ssh.call(op, study='brain-sokoban-v1', python=sys.executable, args=[])


if __name__ == '__main__':
    unittest.main()
