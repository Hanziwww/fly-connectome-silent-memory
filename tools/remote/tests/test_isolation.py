"""The new transport never imports paramiko or the retired tools (tools/remote_study.py, tools/remote_supplement.py,
tools/verify_remote_sync.py), statically and at import time; and it never touches the frozen study's records."""
import ast
import json
import subprocess
import sys
import unittest

from fakes import ROOT

FORBIDDEN = {'paramiko', 'remote_study', 'remote_supplement', 'verify_remote_sync'}
SOURCES = sorted(list((ROOT / 'tools' / 'remote').rglob('*.py')) + list((ROOT / 'studykit').rglob('*.py'))
                 + [ROOT / 'research.py'])
MODULES = ['tools.remote', 'tools.remote.config', 'tools.remote.agent', 'tools.remote.ssh', 'tools.remote.sync',
           'tools.remote.verify_sync', 'tools.remote.queue', 'tools.remote.__main__', 'studykit', 'studykit.registry',
           'studykit.testplan', 'studykit.stats_ext']

TRIPWIRE = r'''
import importlib.abc, json, runpy, sys
sys.path.insert(0, sys.argv[1])
FORBIDDEN = set(json.loads(sys.argv[2]))
attempts = []

class Tripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        parts = set(name.split('.'))
        if parts & FORBIDDEN:
            attempts.append(name)
            raise ImportError('forbidden import: ' + name)
        return None

sys.meta_path.insert(0, Tripwire())
for module in json.loads(sys.argv[3]):
    __import__(module)
runpy.run_path(sys.argv[4], run_name='research_module')  # research.py, without running main()
print(json.dumps(dict(attempts=attempts, loaded=sorted(m for m in sys.modules if set(m.split('.')) & FORBIDDEN))))
'''


def imported_names(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [node.module or ''] + [f'{node.module}.{alias.name}' for alias in node.names]
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', '')
            if name in ('import_module', '__import__', 'run_path', 'run_module', 'spec_from_file_location'):
                names += [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    return names


class Isolation(unittest.TestCase):
    def test_no_static_import_of_retired_transport(self):
        self.assertGreater(len(SOURCES), 10)
        for path in SOURCES:
            for name in imported_names(path):
                parts = set(name.replace('/', '.').replace('\\', '.').split('.'))
                self.assertFalse(parts & FORBIDDEN, f'{path.relative_to(ROOT)} imports {name}')

    def test_no_import_at_run_time(self):
        result = subprocess.run([sys.executable, '-B', '-c', TRIPWIRE, str(ROOT), json.dumps(sorted(FORBIDDEN)),
                                 json.dumps(MODULES), str(ROOT / 'research.py')],
                                capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report, dict(attempts=[], loaded=[]))

    def test_frozen_study_is_protected_everywhere(self):
        from tools.remote import agent, sync
        from studykit import registry
        self.assertIn('brain-sokoban-v1', registry.FROZEN_STUDIES)
        self.assertEqual(set(agent.FROZEN), set(registry.FROZEN_STUDIES))
        self.assertEqual(sync.FROZEN_STUDIES, agent.FROZEN)
        self.assertFalse(agent.remote_owned('studies/brain-sokoban-v1/runs/x/model.pt'))
        self.assertTrue(agent.remote_owned('studies/brain-supplement-v2/runs/x/model.pt'))
        workflow = (ROOT / 'workflow.py').read_text(encoding='utf-8')
        self.assertNotIn('studykit', workflow)


if __name__ == '__main__':
    unittest.main()
