"""Run the tools/remote unit tests (unittest discovery). Network-free: ssh is replaced by tests/fakes.py.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script tools\\remote\\tests\\run_tests.py [pattern]
"""
import sys
import unittest
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(1, str(here.parents[2]))  # the project root, after this folder (for fakes.py)
pattern = sys.argv[1] if len(sys.argv) > 1 else 'test_*.py'
suite = unittest.defaultTestLoader.discover(str(here), pattern=pattern, top_level_dir=str(here))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
