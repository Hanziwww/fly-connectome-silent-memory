"""Finish the closed supplement mirror with a tar stream, then audit every file.

Only transfers saved results. Does not load a model, start a worker, or delete data.
"""
import json
import os
import shlex
import shutil
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import remote_supplement
import verify_remote_sync

transport = remote_supplement.transport
JOBS = ('supp-origin-S2-context-leaky', 'supp-origin-S2-context-neutral-rewired')


def main():
    started = time.time()
    client = transport.connect()
    count = total = 0
    try:
        current = transport.snapshot(client)
        state = current['supervisor']
        assert state['status'] == 'complete' and not state.get('active')
        root, session, _ = transport.settings()
        remote_code = (
            "import json; from pathlib import Path; p=Path(" + repr(session) + ")/'runs'; "
            "files=[f for j in " + repr(JOBS) + " for f in (p/j).rglob('*') "
            "if f.is_file() and not f.name.endswith(('.tmp','.lock'))]; "
            "print(json.dumps({'files':len(files),'bytes':sum(f.stat().st_size for f in files)}))"
        )
        inp, out, err = client.exec_command(shlex.quote(root + '/.venv/bin/python') + ' -B -X utf8 -')
        inp.write(remote_code)
        inp.channel.shutdown_write()
        expected = json.loads(out.read())
        error = err.read().decode('utf-8', errors='replace')
        assert out.channel.recv_exit_status() == 0, error
        assert shutil.disk_usage(HERE).free > expected['bytes'], 'Insufficient local free space for the bounded transfer'
        print(json.dumps({'stream_start': expected, 'jobs': JOBS}), flush=True)

        command = ['tar', '--exclude=*.tmp', '--exclude=*.lock', '-cf', '-', '-C', session + '/runs', *JOBS]
        inp, out, err = client.exec_command(' '.join(shlex.quote(part) for part in command))
        inp.channel.shutdown_write()
        destination = (HERE / 'results').resolve()
        with tarfile.open(fileobj=out, mode='r|', bufsize=1024 * 1024) as archive:
            for member in archive:
                parts = Path(member.name).parts
                assert parts and parts[0] in JOBS and not Path(member.name).is_absolute()
                assert '..' not in parts and not member.issym() and not member.islnk()
                target = (destination / member.name).resolve()
                assert target.is_relative_to(destination)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                assert member.isfile(), member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.download')
                with archive.extractfile(member) as source, temporary.open('wb') as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
                assert temporary.stat().st_size == member.size
                os.replace(temporary, target)
                os.utime(target, (member.mtime, member.mtime))
                count += 1
                total += member.size
                if count % 2000 == 0:
                    print(json.dumps({'streamed_files': count, 'streamed_bytes': total,
                                      'elapsed_seconds': round(time.time() - started, 1)}), flush=True)
        error = err.read().decode('utf-8', errors='replace')
        assert out.channel.recv_exit_status() == 0, error
        assert (count, total) == (expected['files'], expected['bytes'])

        with client.open_sftp() as sftp:
            for name in ('supervisor.json', 'controller.log', 'launch.json',
                         'compatibility-repair.json', 'compatibility-adoption.json',
                         'execution-order-amendment.json'):
                temporary = HERE / (name + '.download')
                sftp.get(session + '/' + name, str(temporary))
                os.replace(temporary, HERE / name)
        latest = transport.snapshot(client)
        assert latest['supervisor']['status'] == 'complete' and not latest['supervisor'].get('active')
        (HERE / 'latest-status.json').write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
        transport.report(latest)
        receipt = dict(method='SSH tar stream; atomic local file replacement', jobs=JOBS,
                       files=count, bytes=total, elapsed_seconds=time.time() - started,
                       training_untouched=True, scientific_acceptance_separate=True)
        (HERE / 'final-transfer.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        print(json.dumps(receipt), flush=True)
    finally:
        client.close()
    return verify_remote_sync.main()


if __name__ == '__main__':
    raise SystemExit(main())
