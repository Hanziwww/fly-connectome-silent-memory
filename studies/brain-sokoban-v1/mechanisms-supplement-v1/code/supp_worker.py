"""One remote GPU worker; baseline acceptance precedes follow-up collection."""
from supp_base import *
import argparse
import itertools
import traceback


def save_condition(out, stem, rows, **metadata):
    features = []; offsets = [0]; stripped = []
    for row in rows:
        features.extend([d['features'] for d in row['decisions']]); offsets.append(len(features))
        item = {k: v for k, v in row.items() if k != 'decisions'}
        item['decisions'] = [{k: v for k, v in d.items() if k != 'features'} for d in row['decisions']]
        stripped.append(item)
    path = out / 'conditions' / (stem + '.npz')
    save_npz(path, features=np.asarray(features, dtype=np.float32).reshape(-1, 512), offsets=np.asarray(offsets))
    write(path.with_suffix('.json'), dict(id=stem, feature_file=path.name, rows=stripped, **metadata))


def preflight(policy, model, task, out):
    from supp_replay import rollout, assert_equal
    from mechanism_model import memory_rollout
    controls = []; count = 0; total = 0
    for delay, prefix in itertools.product((0, 40, 160), (0, 1, 4)):
        reference = read(SOURCE / 'runs' / ('record-' + model['id']) / f'free-after-d{delay}-p{prefix}-behavior.json')
        rows, _ = rollout(policy, task, delay, prefix)
        assert_equal(rows, reference)
        count += sum(r['status'] == 'solved' for r in rows); total += len(rows)
        controls.append(dict(delay=delay, prefix=prefix, source_behavior_matches=True))
        progress(out, stage='reference_acceptance', completed_conditions=len(controls), total_conditions=13)
    assert count == model['source_acceptance']['solved']
    ref = memory_rollout(policy, task, 160, 4)
    recorded, _ = rollout(policy, task, 160, 4, record_path=out/'recorder-pilot.npz')
    assert_equal(recorded, ref)
    groups = read(SESSION/'groups.json')['groups']
    route = dict(indices=groups[0]['indices'], channel='both', strength=0., window='all_decisions')
    sham, _ = rollout(policy, task, 160, 4, route=route); assert_equal(sham, ref)
    restored, _ = rollout(policy, task, 160, 4, spec=dict(kind='restore', part='continuous', fraction=.5, window='hold'))
    assert_equal(restored, ref)
    same, _ = rollout(policy, task, 160, 4, spec=dict(kind='same_cue', part='continuous', window='read'))
    assert_equal(same, ref)
    # A readout-only intervention cannot change the neural dynamics for identical inputs.
    from supp_replay import ordinary_frames
    from mechanism_model import tasklib
    from game import state_from
    import torch
    frames=ordinary_frames(task, 160, 4)
    policy.reset(len(task['cases']))
    for i, frame in enumerate(frames[0]):
        policy.advance_recorded(np.stack([x[i]['board'] for x in frames]), frame['ms'])
    snapshot=policy.encoder.capture_state(detach=True)
    boards=np.stack([tasklib().board(c, state_from(c['level']['start'])) for c in task['cases']])
    q0,f0,_=policy.advance_recorded(boards); original=policy.encoder.capture_state(detach=True)
    policy.encoder.restore_state(snapshot)
    _,_,_=policy.advance_recorded(boards, route=dict(route, channel='readout', strength=1.))
    for name, value in original['tensors'].items():
        assert torch.equal(value, getattr(policy.encoder.brain,name)), 'Direct tap changed the brain'
    # Non-observed cells must have no effect through the direct tap.
    policy.encoder.restore_state(snapshot)
    nonobserved=next(g for g in groups if g['observed_count']==0)
    q1,f1,_=policy.advance_recorded(boards, route=dict(route, indices=nonobserved['indices'], channel='readout',strength=1.))
    np.testing.assert_array_equal(f0,f1); np.testing.assert_array_equal(q0,q1)
    return dict(status='complete', training_solved=count, training_total=total,
        original_solved=count==total, source_acceptance_matches=True, recorder_parity=True,
        zero_gate_and_restore_passed=True, direct_tap_preserves_neural_state=True,
        nonobserved_direct_tap_exact=True, checks=controls)


def selective(policy, model, task, out):
    from supp_replay import rollout, assert_equal
    specs = []
    for window, part, fraction, kind in itertools.product(('write','hold','read'), PARTS, (.25,.5,1.), ('cue','scrambled')):
        specs.append(dict(id=f'{window}-{part}-{fraction}-{kind}',window=window,part=part,fraction=fraction,kind=kind))
    for window, kind in itertools.product(('write','hold','read'), ('same_cue','restore')):
        specs.append(dict(id=f'{window}-{kind}',window=window,part='continuous',fraction=.5,kind=kind))
    done=0; total=len(TIMINGS)*(len(specs)+1); invalid=0
    for delay,prefix in TIMINGS:
        base,_=rollout(policy,task,delay,prefix)
        save_condition(out,f'd{delay}-p{prefix}-baseline',base,delay=delay,prefix=prefix,kind='baseline');done+=1
        for spec in specs:
            rows,audit=rollout(policy,task,delay,prefix,spec=spec)
            if spec['kind'] in ('same_cue','restore'):assert_equal(rows,base)
            if spec['kind']=='scrambled':invalid+=sum(not a['norm_matched'] for a in audit)
            save_condition(out,f'd{delay}-p{prefix}-'+spec['id'],rows,delay=delay,prefix=prefix,spec=spec,audit=audit)
            done+=1;progress(out,stage='selective_state',completed_conditions=done,total_conditions=total,
                            delay=delay,prefix=prefix,condition=spec['id'])
    return dict(status='complete',conditions=done,invalid_norm_matches=invalid,sham_and_restore_passed=True)


def routes(policy, model, task, out):
    from supp_replay import rollout, assert_equal
    groups=[g for g in read(SESSION/'groups.json')['groups'] if g['observed_count']>0]
    specs=[]
    for g,side,strength,window,channel in itertools.product(groups,('indices','controls'),(.25,.5,1.),
            ('first_decision','all_decisions'),('network','readout','both')):
        specs.append(dict(id=f'{g["id"]}-{side}-{strength}-{window}-{channel}',group=g['id'],side=side,
            indices=g[side],strength=strength,window=window,channel=channel))
    done=0;total=len(TIMINGS)*(len(specs)+3)
    for delay,prefix in TIMINGS:
        base,_=rollout(policy,task,delay,prefix)
        save_condition(out,f'd{delay}-p{prefix}-baseline',base,delay=delay,prefix=prefix,kind='baseline');done+=1
        for spec in specs:
            rows,_=rollout(policy,task,delay,prefix,route=spec)
            save_condition(out,f'd{delay}-p{prefix}-'+spec['id'],rows,delay=delay,prefix=prefix,route=spec)
            done+=1;progress(out,stage='route_factorial',completed_conditions=done,total_conditions=total,
                delay=delay,prefix=prefix,condition=spec['id'])
        sham=dict(indices=groups[0]['indices'],strength=0.,window='all_decisions',channel='both')
        rows,_=rollout(policy,task,delay,prefix,route=sham);assert_equal(rows,base)
        save_condition(out,f'd{delay}-p{prefix}-sham',rows,delay=delay,prefix=prefix,kind='sham');done+=1
        rows,_=rollout(policy,task,delay,prefix);assert_equal(rows,base)
        save_condition(out,f'd{delay}-p{prefix}-restore',rows,delay=delay,prefix=prefix,kind='restore');done+=1
    return dict(status='complete',conditions=done,sham_and_restore_passed=True,groups=[g['id'] for g in groups])


def timing_conditions():
    conditions=[]
    for total in (80,120,160,240,320,480,640):
        conditions.append((total,0,'blank'))
        for prefix in (2,4,8):
            if prefix*40>total:continue
            for kind in ('neutral_stationary','neutral_walk_0','neutral_walk_1'):
                conditions.append((total,prefix,kind))
    return conditions


def timing(policy, model, task, out):
    from supp_replay import rollout,timing_frames
    conditions=timing_conditions()
    for done,(total,prefix,kind) in enumerate(conditions,1):
        stem=f't{total}-p{prefix}-{kind}'
        frames=timing_frames(task,total,prefix,kind)
        rows,_=rollout(policy,task,frames=frames,record_path=out/'recordings'/(stem+'.npz'))
        split='held_time' if total not in (80,160,320) else 'held_input' if kind=='neutral_walk_1' else 'discovery'
        save_condition(out,stem,rows,total_postcue_ms=total,prefix=prefix,input_kind=kind,split=split,
            recording='recordings/'+stem+'.npz')
        progress(out,stage='matched_time_inputs',completed_conditions=done,total_conditions=len(conditions),
                 condition=stem,total_postcue_ms=total,split=split)
    return dict(status='complete',conditions=len(conditions),timing_and_input_schedule_locked=True)


def main(job_id):
    plan=read(SESSION/'plan.json')
    assert plan['code_hashes']==code_hashes(), 'Registered supplement code changed'
    for name,checksum in {**plan['source_hashes'],**plan['input_hashes']}.items():
        assert digest(ROOT/name)==checksum, 'Source/input drift: '+name
    job=next(j for j in plan['jobs'] if j['id']==job_id)
    out=SESSION/'runs'/job_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists() or (out/'failure.json').exists():raise RuntimeError('Preserved job cannot be overwritten')
    started=time.time()
    try:
        from supp_replay import FrozenFollowup
        model=get_model(job['model']);task=task_for(model)
        policy=FrozenFollowup(model,out,batch=len(task['cases']));policy.load(checkpoint(model))
        result=globals()[job['kind']](policy,model,task,out)
        audit=policy.frozen_audit()
        result.update(model=model['id'],kind=job['kind'],job=job_id,frozen_audit=audit,parameters_updated=False,
            wall_seconds=time.time()-started,finished_unix=time.time(),pid=os.getpid())
        write(out/'verification.json',dict(status='verified',frozen_audit=audit,
            source_acceptance_matches=result.get('source_acceptance_matches'),
            control_checks_passed=result.get('sham_and_restore_passed',result.get('zero_gate_and_restore_passed')),
            original_solved=result.get('original_solved'),scope='frozen replay; no learning or model selection'))
        write(out/'result.json',result);progress(out,stage='complete',elapsed_seconds=time.time()-started)
        return 0
    except BaseException as exc:
        reason='stopped' if isinstance(exc,InterruptedError) else 'oom' if 'out of memory' in str(exc).lower() else 'numerical' if isinstance(exc,FloatingPointError) else 'implementation_or_acceptance_error'
        write(out/'failure.json',dict(status=reason,error_type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),pid=os.getpid(),wall_seconds=time.time()-started,finished_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    raise SystemExit(main(a.job))
