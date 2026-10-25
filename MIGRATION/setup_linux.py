"""Opt-in Linux dependency installation; never starts training or a cloud queue."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
if os.name == 'nt' or sys.version_info[:2] != (3, 12):
    raise SystemExit('Use CPython 3.12 on Linux. Windows has an included offline runtime.')
venv = root / '.venv-linux'
if venv.exists():
    raise SystemExit('An environment already exists; inspect it rather than overwrite it.')
subprocess.check_call([sys.executable, '-m', 'venv', str(venv)])
python = str(venv / 'bin/python')
subprocess.check_call([python, '-m', 'pip', 'install', 'torch==2.12.1+cu130',
                       '--index-url', 'https://download.pytorch.org/whl/cu130'])
subprocess.check_call([python, '-m', 'pip', 'install', '-r',
                       str(root / 'MIGRATION/requirements-linux.txt')])
subprocess.check_call([python, '-m', 'pip', 'install', '--no-deps', '-e', str(root / 'flybrain')])
print('Environment installed. Run python3.12 MIGRATION/launch.py --probe.')
