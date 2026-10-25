"""Viewer extension for verified joint brain/linear models; preserves legacy experiments."""
from workspace import ARCHIVE, RUNS
import argparse
import json
from http.server import ThreadingHTTPServer

import torch

from bootstrap import HERE
from common import file_hash
from curriculum import encode, legal, restore, solved, state_from, static_dead, step
from readout_models import ActivityLinear
from serve import Handler, Player
from viewer_runtime import ViewerRuntime, decide

RUN = RUNS/'linear-joint/20261002-p03-01'
RUNS = {'joint_linear':RUN,'joint_p05':RUNS/'linear-joint/20261002-p05-01'}


def model_names(task):
    prefix='linear' if task=='P03' else task.lower()
    return dict(zip((f'{prefix}_{suffix}' for suffix in ('joint','source','oldbrain','oldhead','zero')),
        (f'Joint trained · {task}',f'Before {task} training','Restore previous brain',
         'Restore previous readout','Clear brain plasticity')))


MODELS=model_names('P03')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def bundle(path=RUN):
    verification, result, config = read(path/'verification.json'), read(path/'result.json'), read(path/'config.json')
    assert verification['status']=='verified'
    assert file_hash(path/'model.pt')==verification['model_sha256']==result['model_sha256']
    assert file_hash(path/'source.pt')==config['source_checkpoint_sha256']
    return result, verification, config, read(path/'level.json')


def experiments():
    return {suite:(path,bundle(path)) for suite,path in RUNS.items() if (path/'verification.json').exists()}


def checkpoint_ids(models,verification,config):
    final,source,oldbrain,oldhead,zero=models
    return {final:verification['model_sha256'],source:config['source_checkpoint_sha256'],
        oldbrain:verification['model_sha256']+':source-core',
        oldhead:verification['model_sha256']+':source-head',zero:verification['model_sha256']+':zero-core'}


class JointRuntime(ViewerRuntime):
    def __init__(self):
        super().__init__()
        self.heads = {kind:self.head for kind in self.objects}
        for path,(_,verification,config,level) in experiments().values():
            models=model_names(level['id'])
            final = torch.load(path/'model.pt', map_location='cpu', weights_only=True)
            source = torch.load(path/'source.pt', map_location='cpu', weights_only=True)
            variants=[final,source,dict(final,core=source['core']),dict(final,head=source['head']),
                dict(final,core={k:torch.zeros_like(v) for k,v in final['core'].items()})]
            self.objects.update(zip(models,variants))
            self.ids.update(checkpoint_ids(models,verification,config))
            for kind in models:
                head=ActivityLinear().cuda()
                head.load_state_dict(self.objects[kind]['head'],strict=True)
                head.requires_grad_(False)
                self.heads[kind]=head


class JointPlayer(Player):
    def __init__(self):
        super().__init__()
        self.experiments=experiments()
        self.joint_levels={bundle[3]['id']:suite for suite,(_,bundle) in self.experiments.items()}
        for _,(_,_,_,level) in self.experiments.values():
            self.levels[level['id']]=level

    def catalog(self):
        # The legacy catalog only understands its original levels.
        legacy=Player.__new__(Player)
        legacy.__dict__.update(self.__dict__)
        legacy.levels={k:v for k,v in self.levels.items() if k not in self.joint_levels}
        data=Player.catalog(legacy)
        data['joint_experiments'],data['joint_model_details']={},{}
        def public(record):
            return dict(complete=True,solved=int(record['status']=='solved'),total=1,
                        records=[{k:record[k] for k in ('level_id','status','steps')}])
        for suite,(path,(result,verification,config,level)) in self.experiments.items():
            task=level['id']; models=model_names(task)
            final,source,oldbrain,oldhead,zero=models
            data['models'].update(models)
            data['levels'].append(dict({k:v for k,v in level.items() if k!='reference_actions'},
                suite=suite,display_name=f"{task} · {level['boxes']} crates · {level['size']} × {level['size']}",
                descriptor=dict(moves=level['optimal_steps'],pushes=level['minimum_pushes_among_shortest_solutions'])))
            data['suites'][suite]=dict(label=f'{task} · brain + linear',models=list(models),
                default_model=final,reference_model=source,joint=True)
            data['default_suite']=suite
            records={final:result['after_suite']['records'][0],source:result['before_suite']['records'][0],
                oldbrain:result['substitutions']['source_brain_final_readout']['records'][0],
                oldhead:result['substitutions']['final_brain_source_readout']['records'][0],
                zero:result['substitutions']['zero_plasticity_final_readout']['records'][0]}
            for kind,record in records.items():
                data['results'][kind]={suite:dict(test=public(record))}
            data['checkpoint_ids'].update(checkpoint_ids(models,verification,config))
            train=result['training']; length=len(level['reference_actions'])
            rows=[dict(generation=0,current_correct=0,reference_states=length)]
            rows += [dict(generation=p['joint_updates'],current_correct=p['depth'],reference_states=length) for p in train['promotions']]
            completed=length if verification['original_solved'] else max((p['depth'] for p in train['promotions']),default=0)
            rows.append(dict(generation=train['joint_updates'],current_correct=completed,reference_states=length))
            previous=read(path/'previous-level.json')['id']
            summary=dict(task_id=task,size=level['size'],boxes=level['boxes'],previous_task_id=previous,
                training_seconds=train['training_seconds'],updates=train['joint_updates'],seed=config['seed'],
                core_changed=sum(result['checks']['core_changes'].values()),head_changed=sum(result['checks']['head_changes'].values()),
                peak_gib=result['resources']['peak_allocated_gib'],old_task_status=result['old_task_after']['records'][0]['status'],
                old_task_steps=result['old_task_after']['records'][0]['steps'],verified=True)
            training={'Verified suffix completion':rows}
            data['joint_experiments'][suite]=dict(summary=summary,training=training,
                evaluation_path=f'/evidence/{task.lower()}.json')
            details=[f"35,310 brain + 2,052 linear parameters · {train['joint_updates']} joint updates · {train['training_seconds']:.1f} s",
                f'Inherited {previous} brain + linear readout · before {task} learning',
                f'Final linear readout + inherited {previous} brain',f'Final brain + inherited {previous} linear readout',
                'Final linear readout + brain with learned modulation cleared']
            data['joint_model_details'].update(zip(models,details))
            if task=='P03':
                data['joint_summary'],data['joint_training']=summary,training
        return data

    def validate(self,msg):
        if msg.get('level_id') not in self.joint_levels:
            return super().validate(msg)
        source=self.levels[msg['level_id']]
        models=model_names(source['id'])
        kind=msg.get('model',next(iter(models)))
        if kind not in models:
            raise ValueError('Unknown checkpoint for this level')
        obj=msg['state']
        if not isinstance(obj,list) or len(obj)!=2 or type(obj[0]) is not int or not isinstance(obj[1],list) or any(type(x) is not int for x in obj[1]):
            raise ValueError('Invalid state')
        game,state=restore(source),state_from(obj)
        p,boxes=state
        if len(boxes)!=source['boxes'] or len(set(boxes))!=len(boxes) or p in boxes or any(x<0 or x>=100 or x in game['walls'] for x in (p,*boxes)):
            raise ValueError('Invalid placement')
        return source,game,state,kind

    def respond(self,msg,moving):
        source,game,state,kind=self.validate(msg)
        if moving and solved(game,state):
            return dict(terminal=True,state=state)
        if moving and static_dead(game,state):
            return dict(blocked=True,status='proven_corner_deadlock',state=state)
        with self.lock:
            if self.runtime is None:
                self.runtime=JointRuntime()
            runtime=self.runtime
            board,mask=encode(game,state),legal(game,state)
            feature,rates,trace=runtime.response(board,kind)
            joint=source['id'] in self.joint_levels
            reference_model=list(model_names(source['id']))[1] if joint else 'repair_source' if source['id']=='B00' else 'initial'
            reference_kind='before-joint-learning' if joint else 'before-repair' if source['id']=='B00' else 'before-learning'
            before_feature,before_rates,before_trace=runtime.response(board,reference_model)
            reference=dict(rates=before_rates.tolist(),trace=before_trace)
            result=dict(rates=rates.tolist(),trace=trace,source='recorded-simulation',reference=reference,
                reference_kind=reference_kind,checkpoint_id=runtime.ids[kind],reset_per_decision=True,
                reference_label=f"before {source['id']} training" if joint else 'before repair' if source['id']=='B00' else 'before learning',
                readout='linear' if joint else 'legacy-frozen')
            if not mask.any():
                return dict(**result,blocked=True,state=state)
            # Each feature vector is decoded by its own checkpoint's head.
            decision=decide(runtime.heads[kind],feature,mask,runtime.ids[kind])
            before=decide(runtime.heads[reference_model],before_feature,mask,runtime.ids[reference_model])
            reference.update(action=before['action'],probabilities=before['final_probabilities'],
                brain_logits=before['brain_logits'],checkpoint_id=before['checkpoint_id'])
            result.update(brain_logits=decision['brain_logits'],correction_logits=decision['correction_logits'],
                final_probabilities=decision['final_probabilities'],probabilities=decision['final_probabilities'])
            if moving:
                after=step(game,state,decision['action'])
                result.update(action=decision['action'],state=after,terminal=solved(game,after))
            return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8774)
    parser.add_argument('--open',action='store_true')
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.player=JointPlayer()
    if args.open:
        import webbrowser
        webbrowser.open(f'http://127.0.0.1:{args.port}/')
    print(f'FlyLab joint-linear viewer: http://127.0.0.1:{args.port}/',flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
