"""Test doubles for the transport: a fake ssh executor and throwaway projects.

FakeExecutor stands in for OpenSSH. It checks the ssh command line, then runs the real agent (the remote command
that ssh would have run) locally with this interpreter, against a temporary folder playing the remote root. So the
agent's code, its stdin/stdout protocol and the tar streams are all exercised; only the network is mocked.
"""
import base64
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from tools.remote import config as remote_config  # noqa: E402

REMOTE_ROOT = '/srv/flybrain'
GIT_ENV = dict(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_AUTHOR_NAME='test',
               GIT_AUTHOR_EMAIL='test@example.invalid', GIT_COMMITTER_NAME='test',
               GIT_COMMITTER_EMAIL='test@example.invalid')


def make_config(**top):
    data = dict(host='gpu-test', remote_root=REMOTE_ROOT, gpu=dict(lanes=[0], slots_per_lane=2, memory_cap_gib=40),
                disk=dict(min_free_gib=0))
    data.update(top)
    return remote_config.parse(data)


class FakeExecutor:
    """Runs the agent locally instead of over ssh; ``fail`` = (exit code, stderr) simulates an ssh failure."""

    def __init__(self, remote_dir, fail=None):
        self.remote_dir = Path(remote_dir)
        self.fail = fail
        self.calls = []

    def local_argv(self, argv):
        """The remote command of an ssh command line, rewritten to run here against the fake remote root."""
        split = argv.index('--')
        assert len(argv) == split + 3, 'ssh takes exactly the alias and one remote command after --'
        tokens = shlex.split(argv[split + 2])
        params = json.loads(base64.b64decode(tokens[-1]))
        assert params['root'] == REMOTE_ROOT
        params['root'] = str(self.remote_dir)
        if str(params.get('python', '')).startswith(REMOTE_ROOT):
            params['python'] = sys.executable
        tokens[-1] = base64.b64encode(json.dumps(params).encode('utf-8')).decode('ascii')
        return [sys.executable, *tokens[1:]]

    def popen(self, argv, stdin, stdout, stderr):
        self.calls.append(list(argv))
        if self.fail:
            code, message = self.fail
            script = f'import sys; sys.stderr.write({message!r}); sys.exit({code})'
            return subprocess.Popen([sys.executable, '-c', script], stdin=stdin, stdout=stdout, stderr=stderr)
        local = self.local_argv(argv)
        self.remote_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen(local, stdin=stdin, stdout=stdout, stderr=stderr, cwd=str(self.remote_dir))

    def ops(self):
        out = []
        for argv in self.calls:
            tokens = shlex.split(argv[-1])
            out.append(json.loads(base64.b64decode(tokens[-1]))['op'])
        return out


class BannerExecutor(FakeExecutor):
    """Like FakeExecutor, but a 'login script' prints a banner on stdout before the agent runs."""

    BANNER = ('import subprocess, sys; sys.stdout.buffer.write(b"Last login: today\\nWelcome!\\n"); '
              'sys.stdout.flush(); sys.exit(subprocess.call(sys.argv[1:]))')

    def popen(self, argv, stdin, stdout, stderr):
        self.calls.append(list(argv))
        local = self.local_argv(argv)
        self.remote_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.Popen([sys.executable, '-c', self.BANNER, *local], stdin=stdin, stdout=stdout,
                                stderr=stderr, cwd=str(self.remote_dir))


def git(root, *args):
    env = dict(os.environ, **GIT_ENV)
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, env=env)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace'))
    return result.stdout.decode('utf-8', 'replace').strip()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode('utf-8')
    path.write_bytes(data)
    return path


def make_project(base):
    """A small Git project: tracked code, a draft study, an archive folder; untracked data and secrets."""
    root = Path(base) / 'project'
    write(root / '.gitattributes', '* -text\n')
    write(root / '.gitignore', 'data/\nlocal/\nssh.txt\nruntime/\n')
    write(root / 'research.py', 'print("research")\n')
    write(root / 'studykit' / '__init__.py', '"""kit"""\n')
    write(root / 'tools' / 'tool.sh', '#!/bin/sh\necho tool\n')
    write(root / 'archive' / 'old.txt', 'old evidence\n' * 100)
    write(root / 'studies' / 'brain-sokoban-v1' / 'code' / 'registry.py', 'FROZEN = True\n')
    study = root / 'studies' / 'demo-study'
    write(study / 'protocol.json', json.dumps(dict(schema_version=1, study='demo-study', status='draft')) + '\n')
    write(study / 'code' / 'study.py', 'def run(study, args):\n    return 0\n')
    write(study / 'tasks' / 'lock.json', json.dumps(dict(files={})) + '\n')
    write(root / 'data' / 'weights.bin', os.urandom(3 * 2 ** 20 + 17))
    write(root / 'data' / 'nested' / 'table.parquet', os.urandom(4096))
    write(root / 'local' / 'secret.txt', 'never transferred\n')
    write(root / 'ssh.txt', 'never transferred\n')
    git(root, 'init', '-q')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'initial')
    return root


class Workspace:
    """Temporary local project plus a folder that plays the remote root."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.project = make_project(base)
        self.remote = base / 'remote'
        self.executor = FakeExecutor(self.remote)

    def close(self):
        self.tmp.cleanup()
