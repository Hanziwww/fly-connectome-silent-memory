"""OpenSSH transport: every remote operation is one ``ssh <alias> <command>`` subprocess.

The options make a password or host-key prompt impossible: BatchMode=yes, StrictHostKeyChecking=yes, password and
keyboard-interactive authentication off, public keys only. Login therefore uses the key configured for the alias
in ~/.ssh/config (or held by the ssh agent), and the host key must already be in known_hosts. Nothing here reads
~/.ssh, keys or credentials.

The remote half of each operation is tools/remote/agent.py, sent with the command (zlib + base64 in argv) and run
with the remote's stdlib interpreter. ``Executor`` starts the local ssh process; tests substitute a fake executor.
"""
import base64
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import zlib
from pathlib import Path

from . import RemoteError
from .agent import REFUSED, RESULT_MARK, STREAM_MARK

AGENT_SOURCE = Path(__file__).with_name('agent.py').read_bytes()
AGENT_SHA256 = hashlib.sha256(AGENT_SOURCE).hexdigest()
AGENT_ARG = base64.b64encode(zlib.compress(AGENT_SOURCE, 9)).decode('ascii')
BOOTSTRAP = 'import base64,sys,zlib;exec(zlib.decompress(base64.b64decode(sys.argv[1])))'
OPTIONS = ('-T', '-x', '-a',
           '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes',
           '-o', 'PasswordAuthentication=no',
           '-o', 'KbdInteractiveAuthentication=no',
           '-o', 'PreferredAuthentications=publickey',
           '-o', 'ServerAliveInterval=20',
           '-o', 'ServerAliveCountMax=3')
PIPE, DEVNULL = subprocess.PIPE, subprocess.DEVNULL


class Executor:
    """Starts the local OpenSSH client."""

    def popen(self, argv, stdin, stdout, stderr):
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        try:
            return subprocess.Popen(argv, stdin=stdin, stdout=stdout, stderr=stderr, creationflags=flags)
        except FileNotFoundError:
            raise RemoteError(f'OpenSSH client {argv[0]!r} not found; set "ssh" in remote.toml') from None


def _drain(stream, sink):
    for block in iter(lambda: stream.read(65536), b''):
        sink.append(block)
    stream.close()


def _tail(data, limit=2000):
    text = data.decode('utf-8', 'replace').strip()
    return text[-limit:]


class SSH:
    """Remote operations on the host named by ``config`` (a RemoteConfig)."""

    def __init__(self, config, executor=None):
        self.config = config
        self.executor = executor or Executor()

    def argv(self, command):
        """ssh command line: fixed safety options, then '--', the alias and the remote command."""
        executable = shutil.which(self.config.ssh) or self.config.ssh
        return [executable, *OPTIONS, '-o', f'ConnectTimeout={self.config.connect_timeout}', '--', self.config.host,
                command]

    def agent_command(self, op, **params):
        payload = dict(params, op=op, root=self.config.remote_root)
        encoded = base64.b64encode(json.dumps(payload).encode('utf-8')).decode('ascii')
        return shlex.join([self.config.agent_python, '-B', '-X', 'utf8', '-c', BOOTSTRAP, AGENT_ARG, encoded])

    def start(self, op, stdin=DEVNULL, **params):
        return self.executor.popen(self.argv(self.agent_command(op, **params)), stdin=stdin, stdout=PIPE, stderr=PIPE)

    def _check(self, op, code, stdout, stderr):
        if code == 255:
            raise RemoteError(f'ssh to {self.config.host!r} failed (exit 255): {_tail(stderr)}\nCheck the Host entry '
                              f'for {self.config.host!r} in ~/.ssh/config, that its key is loaded, and that the host key '
                              'is already in known_hosts (StrictHostKeyChecking=yes never adds one).')
        if code == REFUSED:
            try:
                reason = json.loads(stderr.decode('utf-8').strip().splitlines()[-1])['refused']
            except (ValueError, KeyError, IndexError):
                reason = _tail(stderr)
            raise RemoteError(f'remote {op} refused: {reason}')
        if code != 0:
            raise RemoteError(f'remote {op} failed (exit {code}): {_tail(stderr)}')

    def _result(self, op, code, stdout, stderr):
        self._check(op, code, stdout, stderr)
        for line in reversed(stdout.decode('utf-8', 'replace').splitlines()):
            if line.startswith(RESULT_MARK):
                return json.loads(line[len(RESULT_MARK):])
        raise RemoteError(f'remote {op} returned no result: {_tail(stdout)} {_tail(stderr)}')

    def call(self, op, input=None, timeout=None, **params):
        """Run one agent operation; ``input``: bytes or a JSON-serialisable value for stdin. Returns its JSON result."""
        if input is not None and not isinstance(input, bytes):
            input = json.dumps(input).encode('utf-8')
        process = self.start(op, stdin=PIPE if input is not None else DEVNULL, **params)
        try:
            stdout, stderr = process.communicate(input, timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RemoteError(f'remote {op} timed out after {timeout} s') from None
        return self._result(op, process.returncode, stdout, stderr)

    def upload(self, op, writer, **params):
        """Run ``op`` with stdin produced by ``writer(stream)`` (e.g. a tar bundle); returns the op's JSON result."""
        process = self.start(op, stdin=PIPE, **params)
        out, err = [], []
        threads = [threading.Thread(target=_drain, args=(process.stdout, out), daemon=True),
                   threading.Thread(target=_drain, args=(process.stderr, err), daemon=True)]
        for t in threads:
            t.start()
        failure = None
        try:
            writer(process.stdin)
        except (BrokenPipeError, ConnectionResetError, OSError) as error:
            failure = error  # the remote side ended early; its stderr says why
        except BaseException:
            process.kill()  # abandon the transfer; an incomplete bundle is never applied
            process.wait()
            for t in threads:
                t.join()
            raise
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
        code = process.wait()
        for t in threads:
            t.join()
        stdout, stderr = b''.join(out), b''.join(err)
        if failure is not None and code == 0:
            raise RemoteError(f'remote {op}: the stream broke ({failure})')
        return self._result(op, code, stdout, stderr)

    def download(self, op, reader, input=None, **params):
        """Run ``op`` and give its stdout, after the agent's stream marker, to ``reader(stream)``; returns its value."""
        if input is not None and not isinstance(input, bytes):
            input = json.dumps(input).encode('utf-8')
        process = self.start(op, stdin=PIPE if input is not None else DEVNULL, **params)
        err = []
        threads = [threading.Thread(target=_drain, args=(process.stderr, err), daemon=True)]
        if input is not None:
            def feed():
                try:
                    process.stdin.write(input)
                except OSError:
                    pass
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
            threads.append(threading.Thread(target=feed, daemon=True))
        for t in threads:
            t.start()
        value, problem = None, None
        try:
            skipped = 0
            while True:  # skip anything a login script printed before the marker
                line = process.stdout.readline()
                if not line:
                    problem = 'the stream ended before its marker'
                    break
                if line == STREAM_MARK:
                    value = reader(process.stdout)
                    break
                skipped += len(line)
                if skipped > 1 << 20:
                    problem = 'no stream marker in the first MiB'
                    break
            while process.stdout.read(65536):
                pass
        finally:
            process.stdout.close()
            code = process.wait()
            for t in threads:
                t.join()
        stderr = b''.join(err)
        self._check(op, code, b'', stderr)
        if problem:
            raise RemoteError(f'remote {op}: {problem}; {_tail(stderr)}')
        return value
