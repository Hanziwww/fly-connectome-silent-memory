"""CPU-only paired outcomes, held-history probes and input-conditioned dynamics."""
from supp_base import *
import pandas as pd


def readout_margin(row, case):
    if not row['decisions']:return None
    first=row['decisions'][0];q=np.asarray(first['logits']);mask=np.asarray(first['legal'],bool)
    action=case['level']['reference_actions'][0]
    others=mask.copy();others[action]=False
    return float(q[action]-q[others].max()) if others.any() else None


def extract():
    records=[]
    for job in read(SESSION/'plan.json')['jobs']:
        out=SESSION/'runs'/job['id']
        if not (out/'result.json').exists() or job['kind']=='preflight':continue
        model=get_model(job['model']);cases={c['id']:c for c in task_for(model)['cases']}
        for path in sorted((out/'conditions').glob('*.json')):
            data=read(path);spec=data.get('spec',{});route=data.get('route',{})
            for case_index,row in enumerate(data['rows']):
                audits=[a for a in data.get('audit',[]) if a['case']==row['case']]
                records.append(dict(model=model['id'],family=model['task'],module=job['kind'],condition=data['id'],
                    case=row['case'],cue=row['cue'],context=row['context'],visible=row['visible'],
                    solved=int(row['status']=='solved'),status=row['status'],steps=row['steps'],
                    first_push_correct=row['first_push_correct'],margin=readout_margin(row,cases[row['case']]),
                    delay=data.get('delay'),prefix=data.get('prefix'),window=spec.get('window',route.get('window')),
                    part=spec.get('part'),fraction=spec.get('fraction'),kind=spec.get('kind',data.get('kind')),
                    group=route.get('group'),side=route.get('side'),strength=route.get('strength'),channel=route.get('channel'),
                    norm_valid=all(a['norm_matched'] for a in audits),
                    total_postcue_ms=data.get('total_postcue_ms'),input_kind=data.get('input_kind'),split=data.get('split'),
                    source=path.relative_to(SESSION).as_posix(),case_index=case_index))
    frame=pd.DataFrame(records)
    if frame.empty:return frame
    baseline=frame[(frame['module']!='timing')&(frame['kind']=='baseline')]
    keys=['model','delay','prefix','case','module']
    lookup=baseline.set_index(keys)[['solved','margin']].to_dict('index')
    for field in ('baseline_solved','baseline_margin','solved_effect','margin_effect'):frame[field]=np.nan
    for i,row in frame[frame['module']!='timing'].iterrows():
        b=lookup[tuple(row[k] for k in keys)]
        frame.loc[i,['baseline_solved','baseline_margin','solved_effect','margin_effect']]=[
            b['solved'],b['margin'],row['solved']-b['solved'],row['margin']-b['margin']]
    return frame


def ridge_decoder(x,y,test):
    mean=x.mean(0);scale=np.maximum(x.std(0),.01)
    a=(x-mean)/scale;b=(test-mean)/scale;center=y.mean()
    coef=a.T@np.linalg.solve(a@a.T+np.eye(len(a)),y-center)
    return b@coef+center


def dynamics(model, series):
    # Discovery data alone fix centering, scaling, PCA and both regressions.
    train=[s for s in series if s['split']=='discovery']
    test=[s for s in series if s['split']!='discovery']
    pool=np.concatenate([s['state'][::4] for s in train])
    if len(pool)>4096:pool=pool[np.linspace(0,len(pool)-1,4096,dtype=int)]
    mean=pool.mean(0);scale=np.maximum(pool.std(0),.01)
    scaled=(pool-mean)/scale
    _,_,vt=np.linalg.svd(scaled,full_matrices=False);basis=vt[:8].T
    for s in series:s['latent']=(s['state']-mean)/scale@basis
    x=[];y=[];u=[]
    for s in train:
        n=len(s['latent'])-1;ids=np.arange(0,n,4)
        x.append(s['latent'][ids]);y.append(s['latent'][ids+1]);u.append(s['input'][ids+1])
    x=np.concatenate(x);y=np.concatenate(y);u=np.concatenate(u)
    simple=np.c_[x,np.ones(len(x))];driven=np.c_[x,u,np.ones(len(x))]
    fit=lambda z:np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@y)
    a=fit(simple);b=fit(driven)
    results=[]
    for split in ('held_input','held_time'):
        blocks=[s for s in test if s['split']==split]
        truth={h:[] for h in (1,20,40,80)};pred={k:{h:[] for h in truth} for k in ('persistence','affine','input_phase')}
        for s in blocks:
            z=s['latent'];last=len(z)-81
            if last<0:continue
            starts=np.unique(np.linspace(0,last,16,dtype=int));p0=z[starts];p1=p0.copy();p2=p0.copy()
            for t in range(1,81):
                p1=np.c_[p1,np.ones(len(starts))]@a
                p2=np.c_[p2,s['input'][starts+t],np.ones(len(starts))]@b
                if t not in truth:continue
                truth[t].append(z[starts+t]);pred['persistence'][t].append(p0.copy())
                pred['affine'][t].append(p1.copy());pred['input_phase'][t].append(p2.copy())
        for h,values in truth.items():
            if not values:continue
            target=np.concatenate(values);denom=((target-target.mean(0))**2).sum()
            for kind in pred:
                estimate=np.concatenate(pred[kind][h]);r2=1-((estimate-target)**2).sum()/max(denom,1e-20)
                results.append(dict(model=model,split=split,horizon_ms=h,method=kind,r2=float(r2),
                    windows=len(target),histories=len(blocks),conditional_on_recorded_inputs=True))
    return results


def probes():
    scores=[];dyn=[]
    for model in models():
        out=SESSION/'runs'/('timing-'+model['id'])
        if not (out/'result.json').exists():continue
        cache=out/'analysis-derived.json'
        # Local derived outputs are kept outside the immutable remote run tree.
        cache=SESSION/'analysis-cache'/(model['id']+'.json')
        manifest=read(out/'manifest.json');key=object_hash(manifest)
        if cache.exists() and read(cache).get('source_manifest')==key:
            value=read(cache);scores.extend(value['probes']);dyn.extend(value['dynamics']);continue
        h=load_pair(checkpoint(model))['head']
        mean=h['mean'].numpy();scale=h['scale'].numpy();items=[];series=[]
        for path in sorted((out/'conditions').glob('*.json')):
            meta=read(path)
            with np.load(path.with_suffix('.npz')) as saved:
                f=saved['features'];offsets=saved['offsets'];first=f[offsets[:-1]]
            z=np.clip((first-mean)/scale,-10,10)
            for i,row in enumerate(meta['rows']):
                items.append(dict(z=z[i],cue=row['cue'],context=row['context'],visible=row['visible'],
                    case=row['case'],condition=meta['id'],split=meta['split'],solved=row['status']=='solved',
                    total_postcue_ms=meta['total_postcue_ms'],input_kind=meta['input_kind']))
            with np.load(out/meta['recording']) as rec:
                latent=rec['instantaneous'];boards=rec['input_boards'];phase=np.eye(4)[rec['phase'].astype(int)]
                for i,row in enumerate(meta['rows']):
                    if row['visible']:continue
                    series.append(dict(state=latent[:,i].astype(np.float64),input=np.c_[boards[:,i],phase],split=meta['split']))
        own=[]
        for visible in (False,True):
            train=[i for i in items if i['split']=='discovery' and i['visible']==visible]
            if not train:continue
            for split in ('held_input','held_time'):
                test=[i for i in items if i['split']==split and i['visible']==visible]
                if not test:continue
                prediction=ridge_decoder(np.stack([i['z'] for i in train]),np.array([i['cue'] for i in train]),np.stack([i['z'] for i in test]))
                for item,prediction_value in zip(test,prediction):
                    own.append(dict(model=model['id'],**{k:v for k,v in item.items() if k!='z'},
                        decoder_score=float(prediction_value),correct=int((prediction_value>=.5)==item['cue']),
                        discovery_histories=len(train)))
        own_dyn=dynamics(model['id'],series)
        write(cache,dict(source_manifest=key,probes=own,dynamics=own_dyn))
        scores.extend(own);dyn.extend(own_dyn)
    return pd.DataFrame(scores),pd.DataFrame(dyn)


def analyze():
    with file_lock(SESSION/'analysis.lock',blocking=False):
        folder=SESSION/'analysis';folder.mkdir(exist_ok=True)
        frame=extract();complete=sum((SESSION/'runs'/j['id']/'result.json').exists() for j in read(SESSION/'plan.json')['jobs'])
        lines=['# Frozen-model supplementary mechanism analysis','',f'Complete jobs included in analysis: {complete}/20.','',
            'Effects are paired by model, history, context, and visibility; histories are not independent training or map replicates.',
            'This supplement follows known earlier results and does not claim real-connectome superiority.','']
        if not frame.empty:
            frame.to_csv(folder/'condition-outcomes.csv',index=False,encoding='utf-8-sig')
            summary=frame.groupby(['model','module','visible','kind','part','fraction','window','side','strength','channel'],dropna=False).agg(
                histories=('solved','size'),solved=('solved','sum'),net_change=('solved_effect','sum'),
                margin_change=('margin_effect','mean'),valid_norm=('norm_valid','all')).reset_index()
            summary.to_csv(folder/'paired-summary.csv',index=False,encoding='utf-8-sig')
            selective=frame[(frame['module']=='selective')&(frame['kind'].isin(['cue','scrambled']))]
            selective.to_csv(folder/'selective-pairs.csv',index=False,encoding='utf-8-sig')
            route=frame[(frame['module']=='routes')&frame['channel'].notna()]
            if not route.empty:
                keys=['model','visible','delay','prefix','case','side','strength','window']
                pivot=route.pivot(index=keys,columns='channel',values='solved_effect').reset_index()
                pivot['interaction']=pivot['both']-pivot['network']-pivot['readout']
                pivot.to_csv(folder/'route-interactions.csv',index=False,encoding='utf-8-sig')
            timing=frame[frame['module']=='timing']
            if not timing.empty:
                tab=timing.groupby(['model','visible','total_postcue_ms','input_kind','prefix'],dropna=False).agg(
                    solved=('solved','sum'),histories=('solved','size')).reset_index()
                tab.to_csv(folder/'matched-time-behavior.csv',index=False,encoding='utf-8-sig')
            for model in MODEL_IDS:
                section=frame[frame['model']==model]
                if section.empty:continue
                lines += [f'## {model}','']
                for visible in sorted(section['visible'].unique()):
                    s=section[section['visible']==visible];label='Visible' if visible else 'Hidden'
                    sel=s[(s['module']=='selective')&(s['window']=='read')&(s['part']=='continuous')&(s['fraction']==.5)]
                    for kind in ('cue','scrambled'):
                        a=sel[sel['kind']==kind]
                        if not a.empty:
                            lost=int(((a['baseline_solved']==1)&(a['solved']==0)).sum());gained=int(((a['baseline_solved']==0)&(a['solved']==1)).sum())
                            lines.append(f'- {label}: pre-read continuous-state {kind} midpoint intervention, full solutions {int(a.solved.sum())}/{len(a)}, baseline {int(a.baseline_solved.sum())}/{len(a)}; lost {lost}, gained {gained}; valid norm controls {int(a.norm_valid.sum())}/{len(a)}.')
                    rr=s[(s['module']=='routes')&(s['side']=='indices')&(s['strength']==1)&(s['window']=='first_decision')]
                    for channel,a in rr.groupby('channel'):
                        lines.append(f'- {label}: first-decision candidate group, {channel}, full solutions {int(a.solved.sum())}/{len(a)}, baseline {int(a.baseline_solved.sum())}/{len(a)}.')
                    tt=s[s['module']=='timing']
                    if not tt.empty:lines.append(f'- {label}: matched-duration inputs, full solutions {int(tt.solved.sum())}/{len(tt)}; see matched-time-behavior.csv for duration/input strata, not a pooled independent-map success rate.')
                lines.append('')
            decode,dyn=probes()
            if not decode.empty:
                decode.to_csv(folder/'held-history-cue-decoding.csv',index=False,encoding='utf-8-sig')
                lines+=['## Interpretive decoding and dynamics','']
                for (model,visible,split),s in decode.groupby(['model','visible','split']):
                    lines.append(f'- {model}, visible={visible}, {split}: cue decoding {int(s.correct.sum())}/{len(s)}; this decoder is not part of the policy.')
            if not dyn.empty:
                dyn.to_csv(folder/'input-conditioned-dynamics.csv',index=False,encoding='utf-8-sig')
                for (model,split),s in dyn[dyn.horizon_ms==40].groupby(['model','split']):
                    description=','.join(f'{r.method} R²={r.r2:.3f}' for r in s.itertuples())
                    lines.append(f'- {model}, {split}, 40 ms conditional prediction: {description}.')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,axes=plt.subplots(1,3,figsize=(15,4.5))
            a=selective[(selective.window=='read')&(selective.part=='continuous')&(selective.fraction==.5)&(~selective.visible)]
            if not a.empty:a.groupby(['model','kind']).solved_effect.mean().unstack().plot.bar(ax=axes[0])
            b=route[(route.side=='indices')&(route.strength==1)&(route.window=='first_decision')&(~route.visible)]
            if not b.empty:b.groupby(['model','channel']).solved_effect.mean().unstack().plot.bar(ax=axes[1])
            if not timing.empty:timing[~timing.visible].groupby(['total_postcue_ms','input_kind']).solved.mean().unstack().plot(ax=axes[2],marker='o')
            for ax,title in zip(axes,['Selective state: paired change','Transmission / direct readout','Matched-time behavior (descriptive)']):
                ax.set_title(title,fontsize=10);ax.tick_params(axis='x',labelsize=7);ax.spines[['top','right']].set_visible(False)
            fig.tight_layout();fig.savefig(folder/'supplement-overview.png',dpi=170);fig.savefig(folder/'supplement-overview.pdf');plt.close(fig)
        lines+=['','## Interpretation limits','',
            'Midpoints remove only operated continuous-variable differences between paired histories; untouched delay buffers may still carry information. Interpret synthetic-state effects with dose, equal-norm controls, visibility, and restoration.',
            'Direct-readout operations test a channel in the artificial observation mapping, not biological neuronal inactivation. Joint effects use within-history paired decomposition.',
            'Successful decoding shows access by the chosen interpretive model; failure does not prove information absent. Held-out low-dimensional predictions are conditional on known inputs and do not establish an attractor.',
            'After all jobs finish, review transfer receipts, paired results, and concrete interpretation before declaring delivery complete; counts alone do not suffice.','']
        (SESSION/'analysis.en.md').write_text('\n'.join(lines),encoding='utf-8')
        result=dict(completed_jobs=complete,total_jobs=20,condition_histories=len(frame),parameters_updated=False,
            all_jobs_present=complete==20,generated_unix=time.time())
        write(SESSION/'analysis.json',result);return result
