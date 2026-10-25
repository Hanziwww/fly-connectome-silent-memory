"""Post-selection generalization, retention and component restoration experiments."""
import json
from pathlib import Path
import os
import time
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'sokoban'))
import torch
from registry import STUDY,SOURCE,read,write
from backend import initialize,set_pair
from behavior import rollouts


def seen_states(out,target_family=None,seen=None):
    exposed=set();visited=set()
    seen=set() if seen is None else seen
    if out.name in seen:return exposed,visited
    seen.add(out.name)
    config=read(out/'config.json')
    from execution import tasks
    current_family=tasks()[config['task']]['family_id']
    include=target_family is None or current_family==target_family
    path=out/'exposure.jsonl'
    if path.exists() and include:
        for line in path.open(encoding='utf-8'):
            for s in json.loads(line)['states']:exposed.add((s[0],tuple(s[1])))
    path=out/'behavior.jsonl'
    if path.exists() and include:
        for line in path.open(encoding='utf-8'):
            for r in json.loads(line)['records']:
                for d in r['decisions']:visited.add((d['state'][0],tuple(d['state'][1])))
    if config.get('parent'):
        parent=STUDY/'runs'/config['parent']
        if (parent/'config.json').exists():
            a,b=seen_states(parent,target_family or current_family,seen);exposed|=a;visited|=b
    return exposed,visited


def evaluate_one(run_id):
    from execution import tasks,stopped
    out=STUDY/'runs'/run_id
    if (out/'generalization.json').exists():return
    if not (out/'model.pt').exists():return
    config=read(out/'config.json');encoder,head=initialize(config,out)
    saved=torch.load(out/'model.pt',map_location='cpu',weights_only=True);set_pair(encoder,head,saved)
    levels=tasks();task=levels[config['task']];check=lambda:stopped(out)
    variants=[l for l in levels.values() if l.get('variant_of')==task['id']]
    novel=[l for l in levels.values() if l['split']=='heldout']
    # Only the difficulty checkpoints enter the zero-shot novel-layout aggregate;
    # transfer-run checkpoints have deliberately trained on a heldout layout.
    if config['module']=='difficulty':variants+=novel
    exposed,visited=seen_states(out,task['family_id'])
    records=rollouts(encoder,head,variants,check)['records'] if variants else []
    for r in records:
        l=levels[r['task']];s=(l['start'][0],tuple(l['start'][1]))
        r.update(split=l['split'],checkpoint_selected_before_test=True,
            start_used_for_training=s in exposed if l['split']=='new_start' else None,
            start_visited_in_training_assessment=s in visited if l['split']=='new_start' else None,
            exposure_scope='current study run and all its study ancestors; earlier P05 lineage not exhaustively reconstructed',
            strict_lifetime_unseen=None if l['split']=='new_start' and s not in exposed|visited else False)
    retention=[]
    if config['module']=='curriculum':
        retention=rollouts(encoder,head,[levels[f'D{i}-T0'] for i in range(task['tier']+1)],check)['records']
    restoration={}
    if config.get('probe_tasks'):
        probes=[levels[t] for t in config['probe_tasks']]
        retention+=rollouts(encoder,head,probes,check)['records']
    origin=config.get('restore_from')
    if config['module']=='parameter_memory' and config['id'].endswith('-B'):
        origin=config['parent']
    if origin and encoder.core is not None:
        old=torch.load(STUDY/'runs'/origin/'model.pt',map_location='cpu',weights_only=True)
        a=levels[read(STUDY/'runs'/origin/'config.json')['task']]
        from game import encode,restore,state_from,legal
        decisions=read(STUDY/'runs'/origin/'after.json')['decisions']
        matched=[]
        if decisions:
            boards=np.stack([encode(restore(a),state_from(d['state'])) for d in decisions])
            masks=np.stack([legal(restore(a),state_from(d['state'])) for d in decisions])
            targets=np.array([d['action'] for d in decisions])
            for name,p in [('current',saved),('A_core_current_head',dict(core=old['core'],head=saved['head'])),
                           ('current_core_A_head',dict(core=saved['core'],head=old['head'])),('A_core_A_head',old)]:
                set_pair(encoder,head,p)
                with torch.no_grad():q=head(torch.as_tensor(encoder.encode(boards,check),device='cuda'))['final_logits'].cpu().numpy()
                competitors=masks.copy();competitors[np.arange(len(targets)),targets]=False
                valid=competitors.any(1);margins=q[np.arange(len(targets)),targets]-np.where(competitors,q,-np.inf).max(1)
                matched.append(dict(pairing=name,original_A_acquired=read(STUDY/'runs'/origin/'result.json').get('final_solved',False),
                    margins=margins[valid].tolist(),mean_margin=float(margins[valid].mean()) if valid.any() else None,
                    decisions=q.tolist(),states=[d['state'] for d in decisions]))
            write(out/'parameter-memory-margins.json',matched)
        for name,p in [('A_core_current_head',dict(core=old['core'],head=saved['head'])),
                       ('current_core_A_head',dict(core=saved['core'],head=old['head'])),
                       ('A_core_A_head',old)]:
            set_pair(encoder,head,p);restoration[name]=rollouts(encoder,head,[a],check)['records'][0]
        # Static parameter-memory consistency control: no learning or simulated
        # sensory input during the wait. Record the actual elapsed wall time.
        from common import cpu_state,tensor_hash
        set_pair(encoder,head,old)
        identity_before=(tensor_hash(cpu_state(encoder.core)),tensor_hash(cpu_state(head)))
        started=time.perf_counter();time.sleep(1.);idle_seconds=time.perf_counter()-started
        identity_after=(tensor_hash(cpu_state(encoder.core)),tensor_hash(cpu_state(head)))
        idle=rollouts(encoder,head,[a],check)['records'][0]
        write(out/'parameter-memory-idle.json',dict(idle_seconds=idle_seconds,
            no_updates_or_simulated_input=True,parameter_hashes_unchanged=identity_before==identity_after,
            saved_A_restoration=restoration['A_core_A_head'],idle_behavior=idle,
            independently_reloaded_A=read(STUDY/'runs'/origin/'verification.json')))
        set_pair(encoder,head,saved)
    write(out/'generalization.json',dict(run_id=run_id,pid=os.getpid(),records=records,
        retention=retention,restoration=restoration,selection_leakage=False))
    encoder.release_graph()


def evaluate_study(run_id=None):
    import subprocess,sys
    from registry import ROOT
    from run_cycle import acquire_worker_lock
    lock=acquire_worker_lock()
    try:
        jobs=read(STUDY/'schedule.json')
        for j in jobs:
            if run_id and run_id!=j['id']:continue
            out=STUDY/'runs'/j['id']
            if not (out/'model.pt').exists() or (out/'generalization.json').exists():continue
            with (out/'evaluation.log').open('a',encoding='utf-8') as log:
                code=subprocess.call([sys.executable,'-B','-X','utf8',str(Path(__file__)),j['id']],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if code:return code
        return 0
    finally:lock.close()


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(STUDY.parents[1]/'sokoban'))
    evaluate_one(sys.argv[1])
