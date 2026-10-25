"""Relocated launcher. Windows uses the included interpreter and package snapshot."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env['PYTHONNOUSERSITE'] = '1'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    paths = [ROOT / 'flybrain', ROOT / 'local/ssh-client']
    if os.name == 'nt':
        interpreter = ROOT / 'MIGRATION/python/python.exe'
        paths.append(ROOT / 'flybrain/.venv/Lib/site-packages')
    else:
        interpreter = ROOT / '.venv-linux/bin/python'
        if not interpreter.is_file():
            raise SystemExit('Run python3.12 MIGRATION/setup_linux.py first.')
    env['PYTHONPATH'] = os.pathsep.join(str(p) for p in paths)
    if args and args[0] == '--probe':
        code = "import sys,json,numpy,torch,flybrain;print(json.dumps(dict(executable=sys.executable,numpy=numpy.__file__,torch=torch.__file__,flybrain=flybrain.__file__,torch_version=torch.__version__,cuda_available=torch.cuda.is_available(),sys_path=sys.path)))"
        command = [str(interpreter), '-B', '-X', 'utf8', '-c', code]
    elif args and args[0] == '--script':
        if len(args) < 2:
            raise SystemExit('--script requires a project-relative Python path')
        target = (ROOT / args[1]).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise SystemExit('Script must exist inside the restored project')
        command = [str(interpreter), '-B', '-X', 'utf8', str(target), *args[2:]]
    else:
        command = [str(interpreter), '-B', '-X', 'utf8', str(ROOT / 'workflow.py'), *args]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
