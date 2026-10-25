"""Formal joint brain/linear workflow. Earlier methods are preserved in archive/."""
import argparse
from pathlib import Path
import runpy
import sys
import unittest


COMMANDS = {
    "serve": ("serve_joint.py", "open the current P05/P03 and historical-model viewer"),
    "train": ("run_linear_next.py", "continue the formal method; requires --task and a new --out"),
    "analyze": ("analyze_linear_joint.py", "summarize a specified joint-training run"),
    "verify": ("verify_joint_http.py", "verify live viewer decisions without training; start serve first"),
    "check": (None, "run the retained CPU method and viewer tests"),
    "study": (None, "run the registered brain-sokoban-v1 research project"),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="; ".join(f"{name}: {description}" for name, (_, description) in COMMANDS.items()),
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="arguments passed to the selected command")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "sokoban"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(directory))
    if args.command == 'study':
        study_code = directory.parent / 'studies' / 'brain-sokoban-v1' / 'code'
        sys.path.insert(0, str(study_code))
        script = study_code / 'study.py'
        sys.argv = [str(script), *args.arguments]
        runpy.run_path(str(script), run_name="__main__")
        return 0
    if args.command == 'check':
        if args.arguments:
            parser.error('check takes no additional arguments')
        suite = unittest.defaultTestLoader.discover(str(directory), pattern='test_*.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    script = directory / COMMANDS[args.command][0]
    sys.argv = [str(script), *args.arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
