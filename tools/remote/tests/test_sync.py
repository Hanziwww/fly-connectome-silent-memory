"""Push and pull through the fake ssh executor: tar round trips, manifests, receipts and refusals."""
import hashlib
import io
import json
import tarfile
import unittest
from pathlib import Path

from fakes import BannerExecutor, Workspace, git, make_config, write

from tools.remote import RemoteError, agent, sync
from tools.remote.ssh import SSH
from studykit.registry import file_lock


def tree_hashes(root, prefix=''):
    root = Path(root)
    return {p.relative_to(root).as_posix(): agent.sha256_file(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).as_posix().startswith(prefix)}


class PushTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.ssh = SSH(make_config(exclude=['archive']), self.ws.executor)
        self.root = self.ws.project

    def tearDown(self):
        self.ws.close()

    def push(self, **kw):
        return sync.push(self.ssh, root=self.root, **kw)

    def test_code_round_trip_matches_the_commit(self):
        sha = git(self.root, 'rev-parse', 'HEAD')
        write(self.root / 'research.py', 'print("uncommitted edit")\n')  # not part of HEAD
        receipt = self.push(data=['data'])
        self.assertTrue(receipt['transport_verified'])
        self.assertEqual(receipt['commit'], sha)
        self.assertIn('research.py', receipt['dirty_tracked_files'])
        remote = self.ws.remote
        self.assertEqual((remote / 'research.py').read_text(), 'print("research")\n')
        self.assertFalse((remote / 'archive').exists(), 'excluded pathspec was pushed')
        for rel in ('data/weights.bin', 'data/nested/table.parquet', 'studykit/__init__.py', 'tools/tool.sh',
                    'studies/brain-sokoban-v1/code/registry.py'):
            self.assertEqual(agent.sha256_file(remote / rel), agent.sha256_file(self.root / rel), rel)
        for secret in ('local/secret.txt', 'ssh.txt'):
            self.assertFalse((remote / secret).exists(), secret)
        current = json.loads((remote / '.sync' / 'current.json').read_text())
        self.assertEqual(current['commit'], sha)
        self.assertIn('data/weights.bin', current['data'])
        self.assertTrue((remote / '.sync' / 'receipts' / f"{receipt['push']}.json").is_file())
        self.assertTrue((self.root / 'runtime' / 'remote' / 'pushes' / f"{receipt['push']}.json").is_file())
        self.assertFalse(list((remote / '.sync' / 'incoming').iterdir()), 'bundle left behind')
        self.assertEqual(self.ws.executor.ops(), ['manifest', 'receive', 'apply'])

    def test_unchanged_data_is_not_resent_and_changed_data_needs_replace(self):
        self.push(data=['data'])
        second = self.push(data=['data'])
        self.assertEqual(second['data_to_send'], [])
        self.assertEqual(second['sent_files'], second['code_files'])
        write(self.ws.remote / 'data' / 'weights.bin', b'remote copy changed')
        with self.assertRaisesRegex(RemoteError, 'differ on the remote'):
            self.push(data=['data'])
        third = self.push(data=['data'], replace_data=True)
        self.assertEqual(third['data_replaced'], ['data/weights.bin'])
        self.assertEqual(agent.sha256_file(self.ws.remote / 'data/weights.bin'),
                         agent.sha256_file(self.root / 'data/weights.bin'))

    def test_secrets_and_machine_state_are_never_sent(self):
        for path in ('local/secret.txt', 'ssh.txt', 'local', 'runtime/x', 'remote.toml', '.git/config',
                     'data/../ssh.txt', '/etc/passwd'):
            with self.subTest(path=path), self.assertRaises(RemoteError):
                self.push(data=[path])
        write(self.root / 'data' / 'id_ed25519', 'key')
        with self.assertRaisesRegex(RemoteError, 'credential'):
            self.push(data=['data'])
        self.assertEqual(self.ws.executor.calls, [], 'a refused push must not contact the remote')

    def test_queue_outputs_cannot_be_pushed(self):
        write(self.root / 'studies' / 'demo-study' / 'runs' / 'job' / 'result.json', '{}')
        with self.assertRaisesRegex(RemoteError, 'remote queue'):
            self.push(data=['studies/demo-study/runs'])

    def test_study_records_must_be_in_the_commit(self):
        with self.assertRaisesRegex(RemoteError, 'schedule.json'):
            self.push(study='demo-study')
        write(self.root / 'studies' / 'demo-study' / 'schedule.json', '[]\n')
        git(self.root, 'add', '-A')
        git(self.root, 'commit', '-q', '-m', 'schedule')
        self.assertTrue(self.push(study='demo-study')['transport_verified'])

    def test_running_queue_blocks_a_push(self):
        self.push()
        lock = self.ws.remote / 'studies' / 'demo-study' / 'worker.lock'
        with file_lock(lock):
            with self.assertRaisesRegex(RemoteError, 'queues are running'):
                self.push()
            self.assertTrue(self.push(allow_running=True)['transport_verified'])

    def test_removed_files_are_reported_stale(self):
        self.push()
        git(self.root, 'rm', '-q', 'tools/tool.sh')
        git(self.root, 'commit', '-q', '-m', 'remove tool')
        receipt = self.push()
        self.assertEqual(receipt['stale_files'], ['tools/tool.sh'])

    def test_dry_run_contacts_only_the_manifest(self):
        summary = self.push(data=['data'], dry_run=True)
        self.assertTrue(summary['dry_run'])
        self.assertEqual(sorted(summary['data_to_send']), ['data/nested/table.parquet', 'data/weights.bin'])
        self.assertEqual(self.ws.executor.ops(), ['manifest'])
        self.assertFalse((self.ws.remote / 'research.py').exists())

    def test_unknown_commit(self):
        with self.assertRaisesRegex(RemoteError, 'not a commit'):
            self.push(commit='no-such-ref')

    def test_tracked_files_are_code_unless_excluded(self):
        with self.assertRaisesRegex(RemoteError, 'already part of the code push'):
            self.push(data=['research.py'])
        receipt = self.push(data=['archive/old.txt'])  # tracked, but excluded from code pushes
        self.assertEqual(receipt['data_to_send'], ['archive/old.txt'])
        self.assertEqual(agent.sha256_file(self.ws.remote / 'archive/old.txt'),
                         agent.sha256_file(self.root / 'archive/old.txt'))

    def test_non_ascii_paths(self):
        write(self.root / 'docs' / 'café notes.md', 'é\n')
        git(self.root, 'add', '-A')
        git(self.root, 'commit', '-q', '-m', 'non-ascii')
        receipt = self.push()
        self.assertTrue(receipt['transport_verified'])
        self.assertEqual((self.ws.remote / 'docs' / 'café notes.md').read_bytes(), 'é\n'.encode())

    def test_failed_bundle_is_abandoned(self):
        self.push()
        sync._archive_members, original = (lambda *a, **k: iter([(None, None)])), sync._archive_members
        try:
            with self.assertRaises(Exception):
                self.push()
        finally:
            sync._archive_members = original
        current = json.loads((self.ws.remote / '.sync' / 'current.json').read_text())
        self.assertEqual(current['commit'], git(self.root, 'rev-parse', 'HEAD'))
        self.assertEqual(self.ws.executor.ops()[-1], 'receive', 'an incomplete bundle must never be applied')


class ApplyRefusals(unittest.TestCase):
    """The remote checks every bundle member itself, whatever the local side sent."""

    def setUp(self):
        self.ws = Workspace()
        self.remote = self.ws.remote

    def tearDown(self):
        self.ws.close()

    def bundle(self, members, manifest=None, name='push-1'):
        buffer = io.BytesIO()
        listed = {}
        with tarfile.open(fileobj=buffer, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for member_name, payload, kind in members:
                info = tarfile.TarInfo(member_name)
                info.type = kind
                if kind == tarfile.SYMTYPE:
                    info.linkname = '/etc/passwd'
                    tar.addfile(info)
                    continue
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
                listed[member_name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
            raw = json.dumps(manifest or dict(commit='c', code=listed, data={}, sent_data=[])).encode()
            info = tarfile.TarInfo(agent.MANIFEST_NAME)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        path = write(self.remote / '.sync' / 'incoming' / f'{name}.tar', buffer.getvalue())
        return agent.sha256_file(path)[1]

    def apply(self, sha, **extra):
        return agent.op_apply(dict(root=str(self.remote), name='push-1', sha256=sha, **extra))

    def test_unsafe_members_are_refused(self):
        cases = [('../evil.py', tarfile.REGTYPE), ('/abs.py', tarfile.REGTYPE), ('a/../../b', tarfile.REGTYPE),
                 ('local/secret.txt', tarfile.REGTYPE), ('ssh.txt', tarfile.REGTYPE), ('x/.ssh/config', tarfile.REGTYPE),
                 ('studies/demo-study/runs/j/result.json', tarfile.REGTYPE), ('studies/demo-study/STOP', tarfile.REGTYPE),
                 ('link.py', tarfile.SYMTYPE)]
        for name, kind in cases:
            with self.subTest(member=name):
                sha = self.bundle([(name, b'payload', kind)])
                with self.assertRaises(agent.Refusal):
                    self.apply(sha)
                self.assertFalse((self.remote / '.sync' / 'incoming' / 'push-1.tar').exists())
        self.assertFalse((self.remote.parent / 'evil.py').exists())

    def test_hash_and_manifest_mismatches_are_refused(self):
        sha = self.bundle([('ok.py', b'print(1)', tarfile.REGTYPE)])
        with self.assertRaisesRegex(agent.Refusal, 'SHA-256'):
            self.apply('0' * 64)
        sha = self.bundle([('ok.py', b'print(1)', tarfile.REGTYPE)],
                          manifest=dict(commit='c', code={}, data={}, sent_data=[]))
        with self.assertRaisesRegex(agent.Refusal, 'manifest'):
            self.apply(sha)

    def test_frozen_study_code_is_accepted(self):
        sha = self.bundle([('studies/brain-sokoban-v1/runs/r/model.pt', b'checkpoint', tarfile.REGTYPE)])
        receipt = self.apply(sha)
        self.assertTrue(receipt['transport_verified'])


class PullTests(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.ssh = SSH(make_config(), self.ws.executor)
        self.root = self.ws.project
        study = self.ws.remote / 'studies' / 'demo-study'
        write(study / 'runs' / 'job-1' / 'result.json', '{"status": "ok"}\n')
        write(study / 'runs' / 'job-1' / 'model.pt', bytes(range(256)) * 4096)
        write(study / 'runs' / 'job-1' / 'run.log', 'line\n')
        write(study / 'runs' / 'job-1' / 'result.json.tmp', 'partial')
        write(study / 'runs' / 'job-2' / 'result.json', '{"status": "running"}\n')
        write(study / 'code-snapshots' / 'abc' / 'manifest.json', '{}\n')
        write(study / 'supervisor.json', json.dumps(dict(status='running', active={'job-2': dict(job='job-2')})))
        write(study / 'events.jsonl', '{"event": "queue_started"}\n')
        self.study = study

    def tearDown(self):
        self.ws.close()

    def test_round_trip_skips_active_jobs_and_transient_files(self):
        receipt = sync.pull(self.ssh, 'demo-study', root=self.root)
        self.assertTrue(receipt['transport_verified'], receipt)
        local = self.root / 'studies' / 'demo-study'
        self.assertEqual(tree_hashes(local, 'runs/job-1'), {k: v for k, v in tree_hashes(self.study).items()
                                                            if k.startswith('runs/job-1') and not k.endswith('.tmp')})
        self.assertFalse((local / 'runs' / 'job-2').exists(), 'an active job was pulled')
        self.assertFalse((local / 'runs' / 'job-1' / 'result.json.tmp').exists())
        skipped = {rel for rel, _ in receipt['skipped']}
        self.assertIn('studies/demo-study/runs/job-2/result.json', skipped)
        self.assertEqual(len(list((local / 'receipts').glob('pull-*.json'))), 1)
        again = sync.pull(self.ssh, 'demo-study', root=self.root)
        self.assertEqual((again['fetched'], again['updated']), ([], []))

    def test_evidence_is_never_overwritten_but_state_mirrors_refresh(self):
        sync.pull(self.ssh, 'demo-study', root=self.root)
        local = self.root / 'studies' / 'demo-study'
        before = (local / 'runs' / 'job-1' / 'result.json').read_bytes()
        write(self.study / 'runs' / 'job-1' / 'result.json', '{"status": "changed"}\n')
        with (self.study / 'events.jsonl').open('a') as f:
            f.write('{"event": "stage_finished"}\n')
        receipt = sync.pull(self.ssh, 'demo-study', root=self.root)
        self.assertFalse(receipt['transport_verified'])
        self.assertEqual(receipt['conflicts'], ['studies/demo-study/runs/job-1/result.json'])
        self.assertEqual((local / 'runs' / 'job-1' / 'result.json').read_bytes(), before)
        self.assertIn('studies/demo-study/events.jsonl', receipt['updated'])
        self.assertEqual((local / 'events.jsonl').read_bytes(), (self.study / 'events.jsonl').read_bytes())

    def test_pull_targets_are_restricted(self):
        for study, paths in (('brain-sokoban-v1', None), ('demo-study', ['studies/demo-study/protocol.json']),
                             ('demo-study', ['studies/other/runs']), ('demo-study', ['studies/demo-study/../x/runs']),
                             ('Bad_Name', None)):
            with self.subTest(study=study, paths=paths), self.assertRaises(RemoteError):
                sync.pull(self.ssh, study, paths=paths, root=self.root)
        self.assertEqual(self.ws.executor.calls, [])

    def test_dry_run(self):
        plan = sync.pull(self.ssh, 'demo-study', root=self.root, dry_run=True)
        self.assertTrue(plan['dry_run'])
        self.assertIn('studies/demo-study/runs/job-1/model.pt', plan['fetch'])
        self.assertFalse((self.root / 'studies' / 'demo-study' / 'runs').exists())

    def test_login_banner_before_the_stream_is_skipped(self):
        executor = BannerExecutor(self.ws.remote)
        receipt = sync.pull(SSH(make_config(), executor), 'demo-study', root=self.root)
        self.assertTrue(receipt['transport_verified'], receipt)
        self.assertEqual(executor.ops(), ['manifest', 'send'])
        self.assertEqual(agent.sha256_file(self.root / 'studies/demo-study/runs/job-1/model.pt'),
                         agent.sha256_file(self.study / 'runs/job-1/model.pt'))


if __name__ == '__main__':
    unittest.main()
