"""Unit tests for msa.manifest: inputs are recorded once; files this run generates are not inputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msa import manifest, paths  # noqa: E402


class GeneratedFiles(unittest.TestCase):
    def setUp(self):
        self.saved = dict(manifest._entries), dict(manifest._hashes)
        manifest._entries.clear()
        manifest._hashes.clear()

    def tearDown(self):
        manifest._entries.clear()
        manifest._hashes.clear()
        manifest._entries.update(self.saved[0])
        manifest._hashes.update(self.saved[1])

    def test_writable_roots_are_generated(self):
        for root in paths.WRITABLE:
            self.assertTrue(manifest.generated(root / 'figure_data' / 'fd_example.csv'))
            self.assertTrue(manifest.generated(root))
        self.assertFalse(manifest.generated(paths.ROOT / 'README.md'))
        self.assertFalse(manifest.generated(paths.ANALYSIS / 'msa' / 'manifest.py'))

    def test_generated_file_is_not_registered(self):
        manifest.register(paths.out('figure_data/fd_example.csv'), b'a,b\n1,2\n')
        self.assertEqual(manifest._entries, {})

    def test_input_is_registered_once(self):
        target = Path(__file__).resolve()
        data = target.read_bytes()
        manifest.register(target, data)
        manifest.register(target, data)
        self.assertEqual(list(manifest._entries), [paths.rel(target)])
        self.assertEqual(manifest._entries[paths.rel(target)]['bytes'], len(data))


if __name__ == '__main__':
    unittest.main()
