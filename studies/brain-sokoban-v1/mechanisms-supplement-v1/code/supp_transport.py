"""Reuse the established SSH client; stream immutable closed jobs with receipts."""
from supp_base import *
import shlex
import tarfile

REMOTE_ROOT='/root/autodl-tmp/brain-sokoban-v1-20261003'
RELATIVE=SESSION.relative_to(ROOT).as_posix()
REMOTE=REMOTE_ROOT+'/'+RELATIVE


def connect():
    sys.path.insert(0,str(ROOT/'tools'))
    from remote_study import connect as established
    return established()


def execute(client,command):
    inp,out,err=client.exec_command(command);inp.channel.shutdown_write()
    result=out.read().decode('utf-8');error=err.read().decode('utf-8')
    if out.channel.recv_exit_status():raise RuntimeError(error or result)
    return result


def snapshot(client):
    return json.loads(execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-B','-X','utf8',REMOTE+'/code/supp_queue.py','status'])))


def report(current):
    s=current['supervisor'];lines=['# Frozen-model mechanism supplement progress','',
        f"Queue: {s.get('status','not_started')}; completed {len(s.get('completed',[]))}/20 jobs.",
        'Five existing models; replay and temporary interventions only. Known partial solutions are recorded separately from technical failures.','',
        '| Job | Status | Progress | GPU |','|---|---|---|---|']
    for row in current['jobs']:
        p=row.get('progress',{});state=row.get('result',row.get('failure',{})).get('status',
            'running' if row.get('actual_process',{}).get('alive') else 'pending')
        lines.append(f"| {row['id']} | {state} | {p.get('stage','')} {p.get('completed_conditions','')}/{p.get('total_conditions','')} | {row.get('process',{}).get('gpu','')} |")
    lines+=['','Local training STOP is retained. History combinations, neurons, and the two S2 orientations are not independent map replicates.','']
    (SESSION/'progress.en.md').write_text('\n'.join(lines),encoding='utf-8')


def deploy(client):
    plan=read(SESSION/'plan.json');assert plan['code_hashes']==code_hashes()
    expected={**plan['source_hashes'],**plan['input_hashes']}
    script='import pathlib,hashlib,json; r=pathlib.Path('+repr(REMOTE_ROOT)+'); expected='+repr(expected)+'; print(json.dumps([p for p,h in expected.items() if not (r/p).exists() or hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]))'
    bad=json.loads(execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-c',script])))
    if bad:raise RuntimeError('Remote source/input mismatch: '+', '.join(bad))
    paths=list(CODE.glob('*.py'))+[SESSION/name for name in ('plan.json','protocol.en.md','models.json','groups.json')]
    with client.open_sftp() as sftp:
        def mkdir(path):
            try:sftp.stat(path)
            except FileNotFoundError:mkdir(path.rsplit('/',1)[0]);sftp.mkdir(path)
        for path in paths:
            dest=REMOTE+'/'+path.relative_to(SESSION).as_posix();mkdir(dest.rsplit('/',1)[0]);sftp.put(str(path),dest)
    verification='import sys;sys.path.insert(0,'+repr(REMOTE+'/code')+');from supp_base import *;assert read(SESSION/"plan.json")["code_hashes"]==code_hashes();print("verified")'
    assert execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-B','-c',verification])).strip()=='verified'
    write(SESSION/'deployment.json',dict(verified=True,files=len(paths),unix=time.time(),remote_root=REMOTE_ROOT))


def sync(client):
    with file_lock(SESSION/'transfer.lock',blocking=False):
        current=snapshot(client);write(SESSION/'latest-status.json',current);report(current)
        with client.open_sftp() as sftp:
            for name in ('supervisor.json','controller.log','storage.json'):
                try:
                    with sftp.open(REMOTE+'/'+name,'rb') as stream:raw=stream.read()
                except FileNotFoundError:continue
                (SESSION/name).write_bytes(raw)
            for row in current['jobs']:
                if row.get('actual_process',{}).get('alive'):continue
                expected=row.get('manifest')
                if not expected:continue
                receipt=SESSION/'receipts'/(row['id']+'.json')
                if receipt.exists() and read(receipt).get('transport_verified'):continue
                check_stop()
                if shutil.disk_usage(SESSION).free-expected['total_bytes']<50*2**30:
                    raise RuntimeError('Local disk reserve would be crossed')
                relative='runs/'+row['id'];dest=SESSION/relative;dest.mkdir(parents=True,exist_ok=True)
                inp,stream,err=client.exec_command(shlex.join(['tar','-C',REMOTE+'/'+relative,'-cf','-','.']))
                inp.channel.shutdown_write()
                with tarfile.open(fileobj=stream,mode='r|') as archive:
                    for member in archive:
                        if member.isdir():continue
                        path=Path(member.name)
                        if not member.isfile() or path.is_absolute() or '..' in path.parts:raise RuntimeError('Unsafe tar member')
                        check_stop();target=dest/path;target.parent.mkdir(parents=True,exist_ok=True)
                        tmp=target.with_name(target.name+'.download')
                        with archive.extractfile(member) as src,tmp.open('wb') as out:shutil.copyfileobj(src,out,4*1024*1024)
                        os.replace(tmp,target)
                error=err.read().decode();code=stream.channel.recv_exit_status()
                if code:raise RuntimeError(error)
                bad=[p for p,d in expected['files'].items() if not (dest/p).is_file() or (dest/p).stat().st_size!=d['bytes'] or digest(dest/p)!=d['sha256']]
                write(receipt,dict(transport_verified=not bad,files=len(expected['files']),bytes=expected['total_bytes'],mismatches=bad,
                    manifest_sha256=object_hash(expected),verified_unix=time.time()))
                if bad:raise AssertionError('Transfer hash mismatch: '+row['id'])
        receipts=[read(p) for p in (SESSION/'receipts').glob('*.json')]
        write(SESSION/'sync-summary.json',dict(verified_jobs=sum(bool(x['transport_verified']) for x in receipts),
            bytes=sum(x['bytes'] for x in receipts if x['transport_verified']),total_jobs=20,unix=time.time()))
        return current


def dispatch(action):
    client=connect()
    try:
        if action=='run':
            try:current=snapshot(client)
            except RuntimeError:current=None
            if current and (current['supervisor'].get('alive') or any(r.get('actual_process',{}).get('alive') for r in current['jobs'])):
                write(SESSION/'latest-status.json',current);report(current);print('Existing queue remains active');return 0
            if current and any('failure' in r for r in current['jobs']):raise RuntimeError('Preserved failure; diagnose without automatic retry')
            deploy(client)
            command=[REMOTE_ROOT+'/.venv/bin/python','-B','-X','utf8','-u',REMOTE+'/code/supp_queue.py','run']
            script='import subprocess,json;f=open('+repr(REMOTE+'/controller.log')+',"a");p=subprocess.Popen('+repr(command)+',cwd='+repr(REMOTE_ROOT)+',stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":p.pid}))'
            value=json.loads(execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-c',script])))
            write(SESSION/'launch.json',dict(**value,unix=time.time(),mode='frozen_only'));print(json.dumps(value));return 0
        current=sync(client) if action=='sync' else snapshot(client)
        write(SESSION/'latest-status.json',current);report(current)
        rows=[{k:v for k,v in r.items() if k in ('id','actual_process','progress','result','verification','failure','process')}
              for r in current['jobs'] if r.get('actual_process',{}).get('alive') or 'result' in r or 'failure' in r]
        print(json.dumps(dict(supervisor=current['supervisor'],gpu=current['gpu'],jobs=rows),ensure_ascii=False));return 0
    finally:client.close()
