"""remote.toml parsing: the example is valid; credentials, bad aliases and bad paths are refused."""
import copy
import tempfile
import tomllib
import unittest
from pathlib import Path

from fakes import ROOT

from tools.remote import ConfigError
from tools.remote import config

EXAMPLE = ROOT / 'tools' / 'remote' / 'remote.toml.example'
BASE = dict(host='pro6000', remote_root='/data/flybrain',
            gpu=dict(lanes=[0], slots_per_lane=2, memory_cap_gib=40, device_memory_gib=96))


def variant(**changes):
    data = copy.deepcopy(BASE)
    for key, value in changes.items():
        if key.startswith('gpu_'):
            data['gpu'][key[4:]] = value
        elif value is None:
            data.pop(key, None)
        else:
            data[key] = value
    return data


class ConfigTests(unittest.TestCase):
    def test_example_parses(self):
        with EXAMPLE.open('rb') as f:
            value = config.parse(tomllib.load(f))
        self.assertTrue(value.remote_root.startswith('/'))
        self.assertTrue(value.python.startswith(value.remote_root))
        self.assertGreaterEqual(value.slots_per_lane, 1)

    def test_defaults_and_paths(self):
        value = config.parse(BASE)
        self.assertEqual(value.python, '/data/flybrain/.venv-linux/bin/python')
        self.assertEqual(value.agent_python, 'python3')
        self.assertEqual(value.slots, [(0, 0), (0, 1)])
        self.assertEqual(value.remote_path('studies', 'x'), '/data/flybrain/studies/x')
        self.assertEqual(value.min_free_disk_gib, 300.0)
        for bad in (('..', 'x'), ('/etc',)):
            with self.subTest(parts=bad), self.assertRaises(ConfigError):
                value.remote_path(*bad)
        self.assertEqual(config.parse(variant(python='/opt/py/bin/python3')).python, '/opt/py/bin/python3')

    def test_credentials_and_connection_details_are_refused(self):
        for key in ('password', 'Password', 'passphrase', 'identity_file', 'IdentityFile', 'private_key', 'token',
                    'user', 'port', 'hostname', 'key', 'proxy_jump'):
            with self.subTest(key=key), self.assertRaisesRegex(ConfigError, 'ssh/config'):
                config.parse(dict(BASE, **{key: 'x'}))
        with self.assertRaisesRegex(ConfigError, 'ssh/config'):
            config.parse(variant(gpu_secret='x'))

    def test_host_must_be_a_plain_alias(self):
        for host in ('user@host', '-oProxyCommand=evil', 'two words', 'host;rm', '', None, 5):
            with self.subTest(host=host), self.assertRaisesRegex(ConfigError, 'alias'):
                config.parse(variant(host=host))

    def test_remote_paths(self):
        for root in ('data/flybrain', '/', '/data/../etc', '/data//x', '/data/./x', '/data/fly brain', '~/x', None):
            with self.subTest(root=root), self.assertRaises(ConfigError):
                config.parse(variant(remote_root=root))
        for python in ('../python', 'bin/py thon', 5):
            with self.subTest(python=python), self.assertRaises(ConfigError):
                config.parse(variant(python=python))

    def test_gpu_settings(self):
        for change in (dict(gpu_lanes=[]), dict(gpu_lanes=[0, 0]), dict(gpu_lanes=[True]), dict(gpu_lanes=[-1]),
                       dict(gpu_slots_per_lane=0), dict(gpu_slots_per_lane=1.5), dict(gpu_memory_cap_gib=0),
                       dict(gpu_memory_cap_gib='40'), dict(gpu_slots_per_lane=3)):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                config.parse(variant(**change))
        data = variant()
        del data['gpu']['memory_cap_gib']
        with self.assertRaisesRegex(ConfigError, 'memory_cap_gib'):
            config.parse(data)

    def test_unknown_keys_and_types(self):
        with self.assertRaisesRegex(ConfigError, 'unknown'):
            config.parse(dict(BASE, lanes=[0]))
        with self.assertRaisesRegex(ConfigError, 'unknown'):
            config.parse(variant(gpu_slots=2))
        for change in (dict(connect_timeout=0), dict(connect_timeout=True), dict(exclude='archive'),
                       dict(exclude=['../x']), dict(agent_python='python3 -c x')):
            with self.subTest(change=change), self.assertRaises(ConfigError):
                config.parse(variant(**change))

    def test_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / 'remote.toml'
            with self.assertRaisesRegex(ConfigError, 'remote.toml.example'):
                config.load(missing)
            missing.write_text('host = "pro6000"\nremote_root = "/data/flybrain"\n[gpu]\nlanes = [0]\n'
                               'memory_cap_gib = 40\n', encoding='utf-8')
            self.assertEqual(config.load(missing).source, str(missing))
            missing.write_text('host = ', encoding='utf-8')
            with self.assertRaises(ConfigError):
                config.load(missing)


if __name__ == '__main__':
    unittest.main()
