"""Remote GPU host commands (run from the project root):

    python -B -X utf8 -m tools.remote check
    python -B -X utf8 -m tools.remote push [--commit HEAD] [--data LIST] [--path P ...] [--study NAME] [--dry-run]
    python -B -X utf8 -m tools.remote pull --study NAME [--path P ...] [--dry-run]
    python -B -X utf8 -m tools.remote verify --study NAME
    python -B -X utf8 -m tools.remote status --study NAME
    python -B -X utf8 -m tools.remote launch --study NAME [--resume] [--clear-stop]
    python -B -X utf8 -m tools.remote stop --study NAME [--reason TEXT]

Settings come from remote.toml at the project root (tools/remote/README.md). Every command is one key-authenticated
ssh call per step; no command ever asks for a password.
"""
import sys
from pathlib import Path

if __package__ in (None, ''):
    # Run as a file (e.g. MIGRATION/launch.py --script tools/remote/__main__.py): import the package from the project
    # root, and keep this folder off sys.path, where queue.py would shadow the standard library's queue module.
    _here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != _here]
    sys.path.insert(0, str(_here.parents[1]))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402

from tools.remote import ConfigError, RemoteError, agent  # noqa: E402
from tools.remote import config as remote_config  # noqa: E402
from tools.remote import sync, verify_sync  # noqa: E402
from tools.remote.ssh import AGENT_SHA256, SSH  # noqa: E402


def queue_arguments(config, resume=False):
    args = ['--lanes', ','.join(str(lane) for lane in config.lanes), '--slots-per-lane', str(config.slots_per_lane),
            '--memory-cap-gib', f'{config.memory_cap_gib:g}', '--gpu-free-reserve-gib',
            f'{config.gpu_free_reserve_gib:g}', '--min-free-disk-gib', f'{config.min_free_disk_gib:g}']
    return args + (['--resume'] if resume else [])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m tools.remote', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', help='settings file (default: remote.toml at the project root)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check', help='connect, report the remote interpreter, disk, GPUs and the last push')
    p = sub.add_parser('push', help='send git archive <commit> and listed data files, hash-verified')
    p.add_argument('--commit', default='HEAD')
    p.add_argument('--data', action='append', default=[], help='file listing project-relative paths, one per line')
    p.add_argument('--path', action='append', default=[], help='project-relative file, folder or glob (repeatable)')
    p.add_argument('--study', help='also check that the commit holds this study\'s protocol, schedule and task lock')
    p.add_argument('--replace-data', action='store_true', help='replace data files that differ on the remote')
    p.add_argument('--allow-running', action='store_true', help='push although a study queue is running')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('pull', help='fetch a study\'s remote outputs that are missing or out of date locally')
    p.add_argument('--study', required=True)
    p.add_argument('--path', action='append', default=[], help='project-relative output path (default: all outputs)')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('verify', help='hash-check the local mirror of an ended study queue')
    p.add_argument('--study', required=True)
    p = sub.add_parser('status', help='remote queue state, STOP, recent events, GPUs and disk')
    p.add_argument('--study', required=True)
    p.add_argument('--events', type=int, default=20)
    p = sub.add_parser('launch', help='start research.py --study NAME run on the remote, detached')
    p.add_argument('--study', required=True)
    p.add_argument('--resume', action='store_true', help='rerun stages that a STOP interrupted')
    p.add_argument('--clear-stop', action='store_true', help='set an existing STOP aside (renamed, kept)')
    p = sub.add_parser('stop', help='create the study\'s STOP file on the remote')
    p.add_argument('--study', required=True)
    p.add_argument('--reason', default='user_requested_pause')
    args = parser.parse_args(argv)
    try:
        config = remote_config.load(args.config)
        ssh = SSH(config)
        if args.command == 'check':
            executable = shutil.which(config.ssh) or config.ssh
            version = subprocess.run([executable, '-V'], capture_output=True, text=True).stderr.strip()
            result = dict(local=dict(ssh=executable, ssh_version=version, agent_sha256=AGENT_SHA256,
                                     settings=config.as_dict()),
                          remote=ssh.call('ping', python=config.python))
        elif args.command == 'push':
            paths = list(args.path)
            for listing in args.data:
                paths += sync.read_list(listing)
            result = sync.push(ssh, commit=args.commit, data=paths, study=args.study, replace_data=args.replace_data,
                               allow_running=args.allow_running, dry_run=args.dry_run)
        elif args.command == 'pull':
            result = sync.pull(ssh, args.study, paths=args.path or None, dry_run=args.dry_run)
        elif args.command == 'verify':
            result = verify_sync.verify(ssh, args.study)
            result = {k: v for k, v in result.items() if k != 'manifest'}
        elif args.command == 'status':
            result = ssh.call('status', study=sync.check_study(args.study, frozen_ok=True), events=args.events)
        elif args.command == 'launch':
            result = ssh.call('launch', study=sync.check_study(args.study), python=config.python,
                              args=queue_arguments(config, args.resume), clear_stop=args.clear_stop)
        else:
            result = ssh.call('stop', study=sync.check_study(args.study), reason=args.reason)
    except (ConfigError, RemoteError, agent.Refusal) as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    ok = result.get('transport_verified', True) if isinstance(result, dict) else True
    return 0 if ok and not (args.command == 'launch' and result.get('exit_code') not in (None, 0)) else 1


if __name__ == '__main__':
    raise SystemExit(main())
