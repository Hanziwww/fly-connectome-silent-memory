"""Use the existing private SSH client; never serialize credentials."""
from mechanism_base import *
import shlex
import tarfile
import stat

REMOTE_ROOT='/root/autodl-tmp/brain-sokoban-v1-20261003'
RELATIVE='studies/brain-sokoban-v1/mechanisms-v1'
REMOTE=REMOTE_ROOT+'/'+RELATIVE


def connect():
    sys.path.insert(0,str(ROOT/'tools'))
    from remote_study import connect as established
    return established()


def execute(client,command):
    stdin,stdout,stderr=client.exec_command(command);stdin.channel.shutdown_write()
    result=stdout.read().decode('utf-8');error=stderr.read().decode('utf-8')
    if stdout.channel.recv_exit_status():raise RuntimeError(error or result)
    return result


def snapshot(client):
    command=[REMOTE_ROOT+'/.venv/bin/python','-B','-X','utf8',REMOTE+'/code/mechanism_queue.py','status']
    return json.loads(execute(client,shlex.join(command)))


def report_status(current):
    s=current['supervisor'];rows=current['jobs']
    lines=['# Frozen-model mechanism study progress','',f"Queue status: {s.get('status','not_started')}; completed {len(s.get('completed',[]))}/{len(rows)} jobs.",
           'All jobs replay or intervene on existing checkpoints; no learning updates.','',
           '| Job | Status | Progress | GPU |','|---|---|---|---|']
    for r in rows:
        p=r.get('progress',{});state=r.get('result',r.get('failure',{})).get('status','running' if r.get('process_alive') else 'pending')
        stage=p.get('stage','');n=p.get('completed_conditions',p.get('completed_states',p.get('completed_histories','')))
        lines.append(f"| {r['id']} | {state} | {stage} {n} | {r.get('process',{}).get('gpu','')} |")
    lines+=['','Technical failures preserve evidence and stop new dispatch; known partial solutions are not technical failures.',
            'Learning groups are discovered on T0 and confirmed on T1; memory groups are discovered on M0, confirmed on S1, and tested for scope on S2.','']
    (SESSION/'progress.en.md').write_text('\n'.join(lines),encoding='utf-8')


def deploy(client):
    plan=read(SESSION/'plan.json')
    assert code_manifest()==plan['code_manifest'] and scientific_sources()==plan['scientific_sources']
    remote_check='import pathlib,hashlib,json; r=pathlib.Path('+repr(REMOTE_ROOT)+'); expected='+repr(dict(plan['scientific_sources'],**plan['backend_data']))+'; print(json.dumps([p for p,h in expected.items() if not (r/p).exists() or hashlib.sha256((r/p).read_bytes()).hexdigest()!=h]))'
    bad=json.loads(execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-c',remote_check])))
    if bad:raise RuntimeError('Remote scientific source differs: '+', '.join(bad))
    paths=list(CODE.glob('*.py'))+list((SESSION/'inputs').rglob('*'))
    paths += [SESSION/n for n in ('plan.json','models.json','registry.json','data-dictionary.json','protocol.en.md') if (SESSION/n).exists()]
    paths=[p for p in paths if p.is_file()]
    with client.open_sftp() as sftp:
        def mkdir(path):
            try:sftp.stat(path)
            except FileNotFoundError:
                mkdir(str(Path(path).parent).replace('\\','/'));sftp.mkdir(path)
        for p in paths:
            remote=REMOTE+'/'+p.relative_to(SESSION).as_posix();mkdir(remote.rsplit('/',1)[0]);sftp.put(str(p),remote)
        # Only extend the old dispatcher; the old training and simulation sources stay intact.
        sftp.put(str(STUDY/'code/study.py'),REMOTE_ROOT+'/studies/brain-sokoban-v1/code/study.py')
    check='from pathlib import Path;import sys;sys.path.insert(0,'+repr(REMOTE+'/code')+');from mechanism_base import *;p=read(SESSION/"plan.json");assert p["code_manifest"]==code_manifest();assert all(digest(SESSION/f)==h for f,h in p["inputs"].items());print("uploaded_inputs_verified")'
    result=execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-B','-c',check]))
    write(SESSION/'deployment.json',dict(verified='uploaded_inputs_verified' in result,files=len(paths),unix=time.time(),remote_root=REMOTE_ROOT))


def copy_small(sftp,relative):
    target=SESSION/relative
    try:
        with sftp.open(REMOTE+'/'+relative,'rb') as f:raw=f.read()
    except FileNotFoundError:return
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.download');tmp.write_bytes(raw);os.replace(tmp,target)


def download_tree(client,relative,expected=None):
    """Stream a closed directory, validate relative paths and file SHA-256."""
    with file_lock(SESSION/'transfer.lock',blocking=False):
        check_stop()
        if expected and shutil.disk_usage(SESSION).free-expected['total_bytes']<50*2**30:
            raise RuntimeError('Local storage reserve would fall below 50 GiB')
        source=REMOTE+'/'+relative;dest=SESSION/relative;dest.mkdir(parents=True,exist_ok=True)
        command=shlex.join(['tar','-C',source,'-cf','-','.'])
        inp,stream,err=client.exec_command(command);inp.channel.shutdown_write()
        with tarfile.open(fileobj=stream,mode='r|') as tar:
            for member in tar:
                if member.isdir():continue
                rel=Path(member.name)
                if not member.isfile() or rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe transfer member')
                target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                if shutil.disk_usage(SESSION).free-member.size<50*2**30:raise RuntimeError('50 GiB local storage reserve reached')
                temp=target.with_name(target.name+'.download')
                with tar.extractfile(member) as src,temp.open('wb') as out:shutil.copyfileobj(src,out,4*1024*1024)
                os.replace(temp,target)
        error=err.read().decode();code=stream.channel.recv_exit_status()
        if code:raise RuntimeError(error)
        if expected:
            bad=[f for f,d in expected['files'].items() if not (dest/f).is_file() or (dest/f).stat().st_size!=d['bytes'] or digest(dest/f)!=d['sha256']]
            receipt=dict(transport_verified=not bad,files=len(expected['files']),bytes=expected['total_bytes'],mismatches=bad,unix=time.time(),manifest_sha256=object_hash(expected))
            write(SESSION/'receipts'/(relative.replace('/','_')+'.json'),receipt)
            if bad:raise AssertionError('Downloaded job does not match remote manifest')


def sync(client):
    current=snapshot(client);write(SESSION/'latest-status.json',current);report_status(current)
    with client.open_sftp() as sftp:
        for name in ('supervisor.json','controller.log','storage.json'):copy_small(sftp,name)
        for row in current['jobs']:
            for name in ('process','progress','result','failure'):
                if name in row:write(SESSION/'runs'/row['id']/(name+'.json'),row[name])
            if row.get('process_alive'):copy_small(sftp,'runs/'+row['id']+'/worker.log')
        if not (SESSION/'neurons/index.npz').exists():
            try:sftp.stat(REMOTE+'/neurons/metadata.json')
            except FileNotFoundError:pass
            else:download_tree(client,'neurons')
    for row in current['jobs']:
        if not ('result' in row or 'failure' in row) or row.get('process_alive'):continue
        rel='runs/'+row['id'];receipt=SESSION/'receipts'/(rel.replace('/','_')+'.json')
        if receipt.exists() and read(receipt)['transport_verified']:continue
        with client.open_sftp() as sftp:
            try:
                with sftp.open(REMOTE+'/'+rel+'/manifest.json','rb') as f:expected=json.loads(f.read())
            except FileNotFoundError:continue
        download_tree(client,rel,expected)
    write(SESSION/'sync-summary.json',dict(unix=time.time(),verified_jobs=len(list((SESSION/'receipts').glob('*.json'))),total_jobs=len(current['jobs'])))
    return current


def dispatch(action,resume=False):
    client=connect()
    try:
        if action=='run':
            # Never deploy while a previous mechanism worker is alive.
            try:current=snapshot(client)
            except RuntimeError:current=None
            if current and (current['supervisor'].get('process_alive') or any(r.get('process_alive') for r in current['jobs'])):
                write(SESSION/'latest-status.json',current);print('Existing remote mechanism queue remains active');return 0
            if current and any('failure' in r for r in current['jobs']):raise RuntimeError('Preserved technical failure requires diagnosis; no automatic restart')
            deploy(client)
            launch='import subprocess,sys,json;f=open('+repr(REMOTE+'/controller.log')+',"a");p=subprocess.Popen('+repr([REMOTE_ROOT+'/.venv/bin/python','-B','-X','utf8','-u',REMOTE+'/code/mechanism_queue.py','run']+(['--resume'] if resume else []))+',cwd='+repr(REMOTE_ROOT)+',stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({"pid":p.pid}))'
            launched=json.loads(execute(client,shlex.join([REMOTE_ROOT+'/.venv/bin/python','-c',launch])))
            write(SESSION/'launch.json',dict(**launched,unix=time.time(),mode='frozen_replay_only'))
            print(json.dumps(launched));return 0
        current=sync(client) if action=='sync' else snapshot(client)
        write(SESSION/'latest-status.json',current);report_status(current)
        compact=dict(supervisor=current['supervisor'],gpu=current['gpu'],jobs=[{k:v for k,v in r.items() if k in ('id','process_alive','progress','failure','result')} for r in current['jobs'] if r.get('process_alive') or 'result' in r or 'failure' in r])
        print(json.dumps(compact,ensure_ascii=False));return 0
    finally:client.close()
