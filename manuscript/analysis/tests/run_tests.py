"""Run all manuscript-analysis unit tests (unittest discovery)."""
import sys
import unittest
from pathlib import Path

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent))
suite = unittest.defaultTestLoader.discover(str(here), pattern='test_*.py', top_level_dir=str(here))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
