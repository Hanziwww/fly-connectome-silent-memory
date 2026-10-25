"""The ssh command line: fixed safety options, no credentials, the agent payload, and error reporting."""
import base64
import json
import shlex
import unittest
import zlib

from fakes import REMOTE_ROOT, FakeExecutor, Workspace, make_config

from tools.remote import RemoteError
from tools.remote.ssh import AGENT_SOURCE, BOOTSTRAP, SSH


def option_values(argv):
    return [argv[i + 1] for i, a in enumerate(argv) if a == '-o']


class CommandLine(unittest.TestCase):
    def setUp(self):
        self.ssh = SSH(make_config(connect_timeout=7), FakeExecutor('unused'))

    def test_options_make_prompts_impossible(self):
        argv = self.ssh.argv('true')
        options = option_values(argv)
        for required in ('BatchMode=yes', 'StrictHostKeyChecking=yes', 'PasswordAuthentication=no',
                         'KbdInteractiveAuthentication=no', 'PreferredAuthentications=publickey', 'ConnectTimeout=7'):
            self.assertIn(required, options)
        self.assertIn('-T', argv)
        self.assertEqual(argv[-3:], ['--', 'gpu-test', 'true'])
        self.assertLess(argv.index('-T'), argv.index('--'))

    def test_no_credentials_on_the_command_line(self):
        argv = self.ssh.argv(self.ssh.agent_command('ping'))
        joined = ' '.join(argv[:-1]).lower()
        for word in ('sshpass', ' -i ', ' -p ', '@', 'identityfile', 'password=yes', 'passphrase'):
            self.assertNotIn(word, joined)
        self.assertFalse(any('password' in o.lower() and not o.endswith('=no') for o in option_values(argv)))

    def test_agent_payload_round_trip(self):
        command = self.ssh.agent_command('manifest', missing_ok=True)
        tokens = shlex.split(command)
        self.assertEqual(tokens[:6], ['python3', '-B', '-X', 'utf8', '-c', BOOTSTRAP])
        self.assertEqual(zlib.decompress(base64.b64decode(tokens[6])), AGENT_SOURCE)
        params = json.loads(base64.b64decode(tokens[7]))
        self.assertEqual(params, dict(op='manifest', root=REMOTE_ROOT, missing_ok=True))
        self.assertEqual(shlex.join(tokens), command)
        self.assertLess(len(command), 30000, 'Windows command lines are limited to 32767 characters')


class Errors(unittest.TestCase):
    def test_ssh_failure_explains_key_and_known_hosts(self):
        ssh = SSH(make_config(), FakeExecutor('unused', fail=(255, 'Permission denied (publickey).')))
        with self.assertRaisesRegex(RemoteError, '(?s)exit 255.*publickey.*known_hosts'):
            ssh.call('ping')

    def test_refusal_and_failure(self):
        ssh = SSH(make_config(), FakeExecutor('unused', fail=(3, '{"refused": "queue running"}\n')))
        with self.assertRaisesRegex(RemoteError, 'refused: queue running'):
            ssh.call('ping')
        ssh = SSH(make_config(), FakeExecutor('unused', fail=(1, 'Traceback: boom')))
        with self.assertRaisesRegex(RemoteError, 'exit 1.*boom'):
            ssh.call('ping')

    def test_agent_refusal_reaches_the_caller(self):
        ws = Workspace()
        try:
            ssh = SSH(make_config(), ws.executor)
            with self.assertRaisesRegex(RemoteError, 'refused: .*frozen'):
                ssh.call('stop', study='brain-sokoban-v1')
            with self.assertRaisesRegex(RemoteError, 'refused: unknown operation'):
                ssh.call('format_disk')
        finally:
            ws.close()


if __name__ == '__main__':
    unittest.main()
