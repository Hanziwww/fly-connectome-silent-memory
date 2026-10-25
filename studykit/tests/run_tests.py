"""Run the studykit unit tests (unittest discovery), including research.py and the study scaffolds.

    .\\MIGRATION\\python\\python.exe -B -X utf8 MIGRATION\\launch.py --script studykit\\tests\\run_tests.py [pattern]
"""
import sys
import unittest
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(1, str(here.parents[1]))  # the project root, after this folder
pattern = sys.argv[1] if len(sys.argv) > 1 else 'test_*.py'
suite = unittest.defaultTestLoader.discover(str(here), pattern=pattern, top_level_dir=str(here))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
