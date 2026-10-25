"""Verify live P03 neural decisions, substitutions and representative legacy models."""
from workspace import ARCHIVE, RUNS
import json
from urllib.request import Request, urlopen

from bootstrap import HERE
from common import save_json
from workspace import RUNTIME

BASE='http://127.0.0.1:8774'
RUN=RUNS/'linear-joint/20261002-p03-01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def request(path,body=None):
    req=Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=90) as response:
        return json.load(response)


def main():
    catalog=request('/api/catalog')
    assert catalog['default_suite'] in catalog['joint_experiments']
    def inspect(value):
        if isinstance(value,dict):
            assert not {'reference_actions','actions','decisions'} & value.keys()
            for child in value.values(): inspect(child)
        elif isinstance(value,list):
            for child in value: inspect(child)
    inspect(catalog)
    from serve_joint import RUNS,model_names
    tasks,results,references=[],{},{}
    for suite in catalog['joint_experiments']:
        result=read(RUNS[suite]/'result.json')
        level_id=read(RUNS[suite]/'level.json')['id']
        final,source,oldbrain,oldhead,zero=model_names(level_id)
        results[level_id]=result; references[level_id]=source
        tasks += [(final,level_id,result['after_suite']['records'][0]),(source,level_id,result['before_suite']['records'][0])]
        for kind,name in [(oldbrain,'source_brain_final_readout'),(oldhead,'final_brain_source_readout'),(zero,'zero_plasticity_final_readout')]:
            tasks.append((kind,level_id,result['substitutions'][name]['records'][0]))
    repair=read(ARCHIVE/'repair/20261001-b00-intervention01/verification.json')
    pilot=read(ARCHIVE/'campaign/run/viewer-verification.json')
    tasks.append(('repair10','B00',repair['results']['repair10']['records'][0]))
    tasks += [('brain',r['level_id'],r) for r in pilot['results']['brain']['records']]
    records=[]
    for kind,level_id,expected in tasks:
        level=next(l for l in catalog['levels'] if l['id']==level_id)
        state,actions,q=level['start'],[],[]
        seen={json.dumps(state)}
        reference=references.get(level_id,'repair_source' if level_id=='B00' else 'initial')
        status='step_timeout'
        for i in range(512):
            response=request('/api/move',dict(level_id=level_id,model=kind,state=state))
            if 'action' not in response:
                status='solved' if response.get('terminal') else response.get('status','blocked')
                break
            assert response['checkpoint_id']==catalog['checkpoint_ids'][kind]
            assert response['trace']['checkpoint_id']==catalog['checkpoint_ids'][kind]
            assert response['reference']['checkpoint_id']==catalog['checkpoint_ids'][reference]
            assert len(response['trace']['events'])==41
            assert response['correction_logits']==[0.,0.,0.,0.]
            if level_id in results and i==0:
                assert response['reference']['brain_logits']==results[level_id]['before_suite']['records'][0]['decisions'][0]['brain_logits']
                assert response['reference_label']==f'before {level_id} training'
            actions.append(response['action']);q.append(response['brain_logits']);state=response['state']
            if response['terminal']:
                status='solved';break
            key=json.dumps(state)
            if key in seen:
                status='repeated_state';break
            seen.add(key)
        assert (status,actions)==(expected['status'],expected['actions']),(kind,status,actions)
        assert q==[d['brain_logits'] for d in expected['decisions']],kind
        records.append(dict(model=kind,level=level_id,status=status,steps=len(actions),actions_and_logits_exact=True))
        print(json.dumps(records[-1]),flush=True)
    # Model switching must not change a cached board's selected or reference head.
    for level_id,result in results.items():
        level=next(l for l in catalog['levels'] if l['id']==level_id)
        restored=request('/api/move',dict(level_id=level_id,model=next(iter(model_names(level_id))),state=level['start']))
        assert restored['brain_logits']==result['after_suite']['records'][0]['decisions'][0]['brain_logits']
    report=dict(status='verified',endpoint=BASE,records=records,
        traced_decisions=sum(r['steps'] for r in records),catalog_has_no_saved_actions=True,
        current_and_reference_readouts_verified=True,model_switching_cache_verified=True,training_performed=False)
    save_json(RUNTIME/'viewer/http-verification.json',report)
    print(json.dumps(dict(status='verified',traced_decisions=report['traced_decisions'])),flush=True)


if __name__=='__main__':
    main()
