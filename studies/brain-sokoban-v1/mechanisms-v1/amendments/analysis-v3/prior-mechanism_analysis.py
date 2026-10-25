"""CPU-only, incremental analysis of saved evidence; never loads a GPU policy."""
from mechanism_base import *
import csv
import pandas as pd


def table(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')


def figure(fig,name):
    import matplotlib.pyplot as plt
    fig.tight_layout();fig.savefig(SESSION/'figures'/(name+'.png'),dpi=180)
    fig.savefig(SESSION/'figures'/(name+'.pdf'));plt.close(fig)


def existing_data():
    from mechanism_prepare import sources
    registry=read(SESSION/'registry.json');curves=[];outcomes=[]
    levels={x['id']:x for x in read(SESSION/'inputs/levels.json')}
    for row in registry:
        folder=ROOT/row['source'];r=read_metadata(folder/'result.json');entry=dict(row)
        level=levels.get(row.get('task'),{})
        for key in ('optimal_steps','boxes','size','family_id','floor_cells','witness_pushes',
                    'floor_branch_cells','articulation_cells','deadlocking_alternatives_on_witness',
                    'shortest_path_count_capped','shortest_path_count_cap'):
            entry[key]=level.get(key)
        entry['retained_result_status']=r.get('status')
        entry['original_solved']=row.get('original_solved')
        outcomes.append(entry)
        p=folder/'history.jsonl'
        if not p.exists():continue
        with p.open(encoding='utf-8') as f:
            for line in f:
                h=json.loads(line);record={k:h.get(k) for k in ('event','depth','updates','elapsed','phase','training_states','loss','bridge_margin','solved','original_status','original_steps','lesson_solved','retained','stage_passed')}
                curves.append(dict(id=row['id'],origin=row['origin'],module=row['module'],task=row['task'],condition=row['condition'],**record))
    table(SESSION/'tables/run-index.csv',outcomes);table(SESSION/'tables/learning-events.csv',curves)
    write(SESSION/'tables/run-index.json',outcomes)
    import matplotlib.pyplot as plt
    difficulty=[r for r in outcomes if r['module']=='difficulty' and r['origin']=='local']
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for row in difficulty:
        tier=int(row['task'][1]);color='#306b83'
        ax[0].scatter(tier,row['training_seconds']/60,color=color)
        ax[1].scatter(row['optimal_steps'],row['updates'],color=color)
    ax[0].set(xlabel='Difficulty tier',ylabel='Local training minutes',xticks=range(8))
    ax[1].set(xlabel='Certified minimum moves (descriptive)',ylabel='Learning updates')
    for a in ax:a.spines[['top','right']].set_visible(False)
    figure(fig,'01-acquisition')
    # Preserve failed controls rather than averaging away their failure modes.
    comparisons=[]
    for d in (1,3,5,7):
        for condition in ('joint','brain_only','head_only','no_recovery','demonstration'):
            selected=[r for r in outcomes if r['task']==f'D{d}-T0' and r['condition']==condition and r['module'] in ('controls','control','difficulty')]
            for row in selected:comparisons.append({k:row.get(k) for k in ('id','origin','task','condition','status','updates','training_seconds','original_solved','parameter_changes')})
    table(SESSION/'tables/training-controls.csv',comparisons)
    return dict(runs=len(outcomes),difficulty=len(difficulty),difficulty_solved=sum(r['original_solved'] is True for r in difficulty),events=len(curves),controls=comparisons)


def parameter_analysis():
    rows=[];planes=[]
    for model in read(SESSION/'models.json'):
        pairs={phase:load_pair(path) for phase,path in unique_phases(model)}
        before=pairs['before'];after=pairs['after']
        for section in ('core','head'):
            for name,tensor in after[section].items():
                old=before[section][name].numpy().astype(float);new=tensor.numpy().astype(float);delta=new-old
                rows.append(dict(model=model['id'],kind=model['kind'],section=section,parameter=name,
                    elements=int(delta.size),changed=int(np.count_nonzero(delta)),l2=float(np.linalg.norm(delta)),
                    maximum_absolute=float(np.max(np.abs(delta))),unit_of_replication='model/type-shared parameter, not neurons'))
        old,_=decision_basis(head_arrays(before));new,s=decision_basis(head_arrays(after))
        principal=np.arccos(np.clip(np.linalg.svd(old.T@new,compute_uv=False),-1,1))*180/np.pi
        planes.append(dict(model=model['id'],old_rank=old.shape[1],new_rank=new.shape[1],
            principal_angles_degrees=principal.tolist(),new_singular_values=s.tolist(),
            standardization_changed=not np.array_equal(before['head']['scale'].numpy(),after['head']['scale'].numpy()) or not np.array_equal(before['head']['mean'].numpy(),after['head']['mean'].numpy())))
    table(SESSION/'tables/parameter-changes.csv',rows);write(SESSION/'tables/readout-planes.json',planes)
    return dict(parameters=len(rows),planes=planes)


def completed_runs():
    for p in sorted((SESSION/'runs').glob('*/result.json')):
        r=read(p)
        receipt=SESSION/'receipts'/('runs_'+p.parent.name+'.json')
        if r.get('status')=='complete' and (os.name!='nt' or (receipt.exists() and read(receipt).get('transport_verified'))):
            yield p.parent,r


def static_analysis():
    rows=[];atlas=[];annotations=None
    path=SESSION/'neurons/annotations.parquet'
    if path.exists():annotations=pd.read_parquet(path)
    for folder,r in completed_runs():
        if r.get('kind')!='static' or not (folder/'decomposition.npz').exists():continue
        bank=read(folder/'bank.json');targets=np.array([x['action'] for x in bank['labels']])
        with np.load(folder/'decomposition.npz') as z:
            latest=z['q11'].copy();latest[np.arange(len(targets)),targets]=-np.inf;competitor=latest.argmax(1)
            for name in ('brain','head','interaction'):
                margin=z[name][np.arange(len(targets)),targets]-z[name][np.arange(len(targets)),competitor]
                rows.append(dict(model=r['model'],contribution=name,mean_margin_change=float(margin.mean()),
                    median_margin_change=float(np.median(margin)),teacher_competitor='final strongest alternative, fixed across four combinations',
                    reconstruction_error=r['decomposition_error']))
        if annotations is None:continue
        summary=[]
        for phase in r['phases']:
            with np.load(folder/f'matched-{phase}.npz') as a:
                mean=a['mean'];event=np.array([int(x['push']) for x in bank['labels']]);actions=targets
                stats={}
                for k,field in enumerate(FIELDS):
                    stats[field+'_mean']=mean[:,k].mean(0)
                    stats[field+'_across_state_sd']=mean[:,k].std(0)
                    if 0<event.sum()<len(event):stats[field+'_push_contrast']=mean[event==1,k].mean(0)-mean[event==0,k].mean(0)
                for action in range(4):
                    if (actions==action).any():stats[f'v_action{action}_mean']=mean[actions==action,2].mean(0)
                    safe=np.array([x['solvable_actions'][action] for x in bank['labels']])
                    if safe.any() and (~safe).any():stats[f'v_solvable_action{action}_contrast']=mean[safe,2].mean(0)-mean[~safe,2].mean(0)
                frame=annotations[['bodyId','parameter_group','direct_input','observed_by_policy']].copy()
                for name,value in stats.items():frame[name]=value
                frame['model']=r['model'];frame['phase']=phase
                target=SESSION/'atlas'/f'{r["model"]}-{phase}.parquet';target.parent.mkdir(exist_ok=True);frame.to_parquet(target,index=False)
                if 'type' in annotations:
                    frame['type']=annotations['type'].fillna('unannotated')
                    for name,group in frame.groupby('type',sort=False):
                        summary.append(dict(model=r['model'],phase=phase,type=name,cells=len(group),
                            spike_probability=float(group.spikes_mean.mean()),voltage=float(group.v_mean.mean()),
                            adaptation=float(group.adapt_mean.mean()),resource=float(group.res_mean.mean())))
        atlas.extend(summary)
    table(SESSION/'tables/brain-head-decomposition.csv',rows)
    if atlas:pd.DataFrame(atlas).to_parquet(SESSION/'tables/type-functional-atlas.parquet',index=False)
    if rows:
        import matplotlib.pyplot as plt
        frame=pd.DataFrame(rows).pivot(index='model',columns='contribution',values='mean_margin_change')
        fig,ax=plt.subplots(figsize=(11,max(4,.25*len(frame))))
        frame.plot.barh(ax=ax,color=['#387b98','#bf8754','#706292']);ax.set_xlabel('Paired change in fixed action margin');figure(fig,'02-brain-readout')
    if atlas:
        frame=pd.DataFrame(atlas);frame=frame.groupby(['model','phase'])[['spike_probability','adaptation','resource']].mean()
        fig,ax=plt.subplots(1,3,figsize=(12,4))
        for i,col in enumerate(frame):
            x=frame[col].unstack('phase');x=x.reindex(columns=[v for v in ('before','mid','after') if v in x.columns])
            ax[i].plot(np.arange(len(x.columns)),x.T,alpha=.25,color='#387b98');ax[i].set(xticks=range(len(x.columns)),xticklabels=x.columns,title=col)
        figure(fig,'03-neuron-atlas')
    return dict(decomposition_rows=len(rows),type_summary_rows=len(atlas))


def pca_fit(x,components=8):
    mean=x.mean(0);scale=x.std(0);scale=np.maximum(scale,.01)
    z=(x-mean)/scale;cov=z.T@z/max(1,len(z)-1);values,vectors=np.linalg.eigh(cov)
    order=np.argsort(values)[::-1][:min(components,len(x)-1)]
    return mean,scale,vectors[:,order],values[order]


def ridge_decode(x,y,test):
    mean=x.mean(0);scale=np.maximum(x.std(0),.01);x=(x-mean)/scale;test=(test-mean)/scale
    centered=y-y.mean();w=x.T@np.linalg.solve(x@x.T+np.eye(len(x)),centered)
    return test@w+y.mean()


def factor_model(train,held,out):
    """Regularized reduced-rank demixing, with cue/context/time marginalizations."""
    contexts=sorted({e['context'] for e in train});mean=np.mean(np.concatenate([e['decision'] for e in train]),0)
    scale=np.maximum(np.std(np.concatenate([e['decision'] for e in train]),0),.01)
    def tensor(rows):
        return np.stack([np.stack([np.mean([e['decision'] for e in rows if e['cue']==cue and e['context']==context],0)
                                 for context in contexts]) for cue in (0,1)])
    x=(tensor(train)-mean)/scale;h=(tensor(held)-mean)/scale
    # The centered data tensor has axes cue x context x time x feature.
    grand=x.mean((0,1,2),keepdims=True);x=x-grand;h=h-grand
    marginals=dict(time=x.mean((0,1),keepdims=True),cue_time=x.mean(1,keepdims=True)-x.mean((0,1),keepdims=True))
    if len(contexts)>1:
        marginals['context_time']=x.mean(0,keepdims=True)-x.mean((0,1),keepdims=True)
        marginals['cue_context_time']=x-x.mean(0,keepdims=True)-x.mean(1,keepdims=True)+x.mean((0,1),keepdims=True)
    X=x.reshape(-1,512).T;H=h.reshape(-1,512).T
    inv=np.linalg.solve(X.T@X+np.eye(X.shape[1]),X.T)
    report=[];arrays={'mean':mean,'scale':scale,'grand':grand,'train_tensor':x.astype('float32'),'held_tensor':h.astype('float32')}
    for name,part in marginals.items():
        Y=np.broadcast_to(part,x.shape).reshape(-1,512).T;B=Y@inv
        fitted=B@X;U,_,_=np.linalg.svd(fitted,full_matrices=False);U=U[:,:min(3,np.linalg.matrix_rank(fitted))]
        encoder=U.T@B;prediction=U@encoder@H
        # Held-out marginal target is computed only for evaluation.
        target={'time':h.mean((0,1),keepdims=True),'cue_time':h.mean(1,keepdims=True)-h.mean((0,1),keepdims=True),
            'context_time':h.mean(0,keepdims=True)-h.mean((0,1),keepdims=True),
            'cue_context_time':h-h.mean(0,keepdims=True)-h.mean(1,keepdims=True)+h.mean((0,1),keepdims=True)}[name]
        target=np.broadcast_to(target,h.shape).reshape(-1,512).T
        score=float(1-np.square(prediction-target).sum()/max(np.square(target).sum(),1e-12))
        report.append(dict(factor=name,components=U.shape[1],heldout_reconstruction_r2=score,regularization=1.0))
        arrays[name+'_encoder']=encoder;arrays[name+'_decoder']=U
    save_npz(out,**arrays)
    return report


def memory_analysis():
    rows=[];model_scores=[];dynamics=[];decode=[];demixing=[]
    for folder,result in completed_runs():
        if result.get('kind')!='memory' or not (folder/'matched-index.json').exists():continue
        model=model_by_id(result['model']);idx=read(folder/'matched-index.json');heads={p:head_arrays(load_pair(path)) for p,path in unique_phases(model)}
        episodes=[]
        for entry in idx:
            phase=entry['phase'];h=heads[phase];basis,_=decision_basis(h)
            with np.load(folder/entry['readout']) as f:feature=f['features'][-1];q=f['logits'][-1]
            cases=read(folder/Path(entry['activity']).with_suffix('.json'))['cases']
            z=standardized(feature,h)
            for i,c in enumerate(cases):
                if c['cue']!=0:continue
                j=next(k for k,d in enumerate(cases) if d['cue']==1 and d['context']==c['context'] and d['visible']==c['visible'])
                delta=z[j].astype(float)-z[i];length=float(delta@delta);projected=delta@basis
                rows.append(dict(model=model['id'],phase=phase,delay=entry['delay'],prefix=entry['prefix'],context=c['context'],visible=c['visible'],
                    cue_distance=float(length**.5),readout_energy_fraction=float(projected@projected/length) if length>1e-16 else None,
                    centered_logit_contrast=float(np.linalg.norm((q[j]-q[i])-(q[j]-q[i]).mean())),
                    split='train' if entry['delay'] in (0,40,160) and entry['prefix'] in (0,1,4) else 'heldout'))
            if entry['delay']==160 and entry['prefix']==4:
                with np.load(folder/entry['activity']) as f:
                    states=f['mean'];rowdata={}
                    for ci,c in enumerate(cases):
                        for k,field in enumerate(FIELDS):rowdata[f"case{ci}_{field}"]=states[-1,ci,k]
                    save_npz(SESSION/'atlas'/f'{model["id"]}-{phase}-memory.npz',**rowdata)
            if phase=='after':
                with np.load(folder/entry['activity']) as f:
                    inst=f['instantaneous'];last=inst[:,-40:]
                    for k,c in enumerate(cases):episodes.append(dict(**c,delay=entry['delay'],prefix=entry['prefix'],
                        split='train' if entry['delay'] in (0,40,160) and entry['prefix'] in (0,1,4) else 'heldout',instant=inst[k],decision=last[k]))
        train=[e for e in episodes if e['split']=='train'];held=[e for e in episodes if e['split']=='heldout']
        for factor in factor_model(train,held,SESSION/'tables'/f'{model["id"]}-demixed.npz'):
            demixing.append(dict(model=model['id'],**factor))
        # Coordinates and decoders are fit on training histories only, never on outcomes.
        mean,scale,basis,eigen=pca_fit(np.concatenate([e['instant'] for e in train]))
        coords=[(e,(e['instant']-mean)/scale@basis) for e in episodes]
        save_npz(SESSION/'tables'/f'{model["id"]}-shared-pca.npz',mean=mean,scale=scale,basis=basis,eigenvalues=eigen)
        x=np.concatenate([z[:-1] for e,z in coords if e['split']=='train']);y=np.concatenate([z[1:] for e,z in coords if e['split']=='train'])
        # Affine observational LDS, with exogenous input phases not conditioned out.
        design=np.column_stack((x,np.ones(len(x))));coef=np.linalg.solve(design.T@design+np.eye(design.shape[1]),design.T@y)
        hx=np.concatenate([z[:-1] for e,z in coords if e['split']=='heldout']);hy=np.concatenate([z[1:] for e,z in coords if e['split']=='heldout'])
        prediction=np.column_stack((hx,np.ones(len(hx))))@coef
        r2=float(1-np.square(hy-prediction).sum()/max(np.square(hy-hy.mean(0)).sum(),1e-12))
        eigenA=np.linalg.eigvals(coef[:-1]);radii=np.abs(eigenA)
        # Finite-horizon singular values measure transient propagation in the fitted coordinate system.
        gain=[float(np.linalg.svd(np.linalg.matrix_power(coef[:-1],t),compute_uv=False)[0]) for t in (1,20,40,80,160)]
        dynamics.append(dict(model=model['id'],heldout_r2=r2,accepted_as_predictive=r2>0,
            eigenvalue_real=eigenA.real.tolist(),eigenvalue_imag=eigenA.imag.tolist(),
            stable_decay_ms=[float(-1/np.log(v)) if 0<v<1 else None for v in radii],
            propagation_horizons_ms=[1,20,40,80,160],max_singular_gain=gain,
            interpretation='Descriptive affine dynamics; input-driven trajectories do not establish an attractor.'))
        trainx=np.stack([e['decision'] for e in train]);testx=np.stack([e['decision'] for e in held]);y=np.array([e['cue'] for e in train]);ty=np.array([e['cue'] for e in held])
        matrix=np.empty((40,40))
        for t in range(40):
            pred=ridge_decode(trainx[:,t],y,testx.reshape(-1,512)).reshape(len(testx),40)
            matrix[t]=((pred>=.5)==ty[:,None]).mean(0)
        save_npz(SESSION/'tables'/f'{model["id"]}-time-generalization.npz',accuracy=matrix,
            samples_train=np.array(len(train)),samples_held=np.array(len(held)))
        decode.append(dict(model=model['id'],same_time_mean_accuracy=float(np.diag(matrix).mean()),
            all_time_mean_accuracy=float(matrix.mean()),unit='histories within one family',input='instantaneous pooled current/voltage; not actual mixed-time policy features'))
        for behavior in folder.glob('free-after-*-behavior.json'):
            for r in read(behavior):model_scores.append(dict(model=model['id'],delay=r['delay_ms'],prefix=r['prefix_windows'],
                context=r['context'],cue=r['cue'],visible=r['visible'],split=r['history_split'],status=r['status'],
                first_push_correct=r['first_push_correct'],steps=r['steps']))
    table(SESSION/'tables/memory-geometry.csv',rows);table(SESSION/'tables/memory-behavior.csv',model_scores)
    write(SESSION/'tables/dynamics.json',dynamics);write(SESSION/'tables/time-decoders.json',decode)
    write(SESSION/'tables/demixed-reconstruction.json',demixing)
    if rows:
        import matplotlib.pyplot as plt
        df=pd.DataFrame(rows);df=df[(df.phase=='after')&~df.visible]
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for model,part in df.groupby('model'):
            data=part.groupby('delay')[['cue_distance','readout_energy_fraction']].mean()
            axes[0].plot(data.index,data.cue_distance,'o-',label=model.replace('remote-dynamic-','').replace('supp-',''))
            axes[1].plot(data.index,data.readout_energy_fraction,'o-')
        axes[0].set(xlabel='Delay (ms)',ylabel='Standardized cue distance');axes[1].set(xlabel='Delay (ms)',ylabel='Energy in centered readout row space')
        axes[0].legend(fontsize=7);figure(fig,'04-memory-geometry')
    return dict(geometry_rows=len(rows),behavior_histories=len(model_scores),dynamics=dynamics,decoders=decode,demixing=demixing)


def intervention_analysis():
    rows=[];paths=[]
    for folder,result in completed_runs():
        if (folder/'effects.json').exists():
            for r in read(folder/'effects.json'):
                if 'paired' in r:
                    for p in r['paired']:rows.append(dict(job=folder.name,model=result['model'],condition=r['id'],delay=r['delay'],prefix=r['prefix'],**p))
                else:rows.append(dict(job=folder.name,model=result['model'],condition=r['id'],solved_effect=int(r['solved'])-int(r['baseline_solved']),**r))
        if (folder/'pathways.json').exists():paths=read(folder/'pathways.json')['rows']
    table(SESSION/'tables/causal-effects.csv',rows);table(SESSION/'tables/signed-pathways.csv',paths)
    import matplotlib.pyplot as plt
    if rows:
        df=pd.DataFrame(rows);pivot=df.groupby(['model','condition']).solved_effect.mean().unstack(0)
        fig,ax=plt.subplots(figsize=(11,max(5,.12*len(pivot))));im=ax.imshow(pivot.fillna(np.nan),aspect='auto',vmin=-1,vmax=1,cmap='RdBu')
        ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=45,ha='right',fontsize=7)
        ax.set_yticks(range(len(pivot)),pivot.index,fontsize=5);fig.colorbar(im,ax=ax,label='Paired solution change; descriptive histories');figure(fig,'05-causal-interventions')
    if paths:
        df=pd.DataFrame(paths);fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(df))
        ax.bar(x,df.direct_absolute,color='#397b95',label='Direct');ax.bar(x,df.two_hop_absolute,bottom=df.direct_absolute,color='#c9985c',label='Two-hop')
        ax.set(yscale='symlog',ylabel='Structural transfer mass (different path lengths)',xticks=x,xticklabels=df.group+' '+df.side)
        ax.tick_params(axis='x',labelrotation=80,labelsize=6);ax.legend();figure(fig,'06-signed-pathways')
    return dict(causal_rows=len(rows),path_rows=len(paths))


def cross_family_analysis():
    """M0 and T0 fitted explanations, evaluated on untouched S1/S2 and T1 families."""
    available={r.get('model'):(p,r) for p,r in completed_runs() if r.get('model')}
    model='remote-dynamic-M0-head_only';memory=[];learning=[];anatomy=[]
    if model in available:
        root,_=available[model];train=[];targets=[]
        for row in read(root/'matched-index.json'):
            if row['phase']!='after' or row['delay'] not in (0,40,160) or row['prefix'] not in (0,1,4):continue
            with np.load(root/row['readout']) as f:train.extend(f['features'][-1])
            targets.extend([c['cue'] for c in read(root/Path(row['activity']).with_suffix('.json'))['cases']])
        train=np.asarray(train);targets=np.asarray(targets);mean,scale,basis,eigen=pca_fit(train)
        save_npz(SESSION/'tables/M0-fixed-cross-family-pca.npz',mean=mean,scale=scale,basis=basis,eigenvalues=eigen)
        for name in ('remote-dynamic-M0-head_only','supp-S1-ring-brain','supp-S2-context-brain'):
            if name not in available:continue
            folder,_=available[name]
            for row in read(folder/'matched-index.json'):
                if row['phase']!='after':continue
                with np.load(folder/row['readout']) as f:test=f['features'][-1]
                cases=read(folder/Path(row['activity']).with_suffix('.json'))['cases']
                prediction=ridge_decode(train,targets,test);coords=(test-mean)/scale@basis
                for i,c in enumerate(cases):memory.append(dict(discovery_model=model,evaluation_model=name,
                    family=model_by_id(name)['task'],context=c['context'],cue=c['cue'],visible=c['visible'],delay=row['delay'],prefix=row['prefix'],
                    predicted_cue=int(prediction[i]>=.5),decoded_correct=bool((prediction[i]>=.5)==c['cue']),
                    fit_on_this_family=name==model,heldout_history=row['delay'] not in (0,40,160) or row['prefix'] not in (0,1,4),
                    pc1=float(coords[i,0]),pc2=float(coords[i,1])))
    # Generalization of task-state information, with raw board features as a paired comparator.
    import bootstrap
    from game import encode,restore,state_from
    levels={l['id']:l for l in read(SESSION/'inputs/levels.json')}
    for phase in ('before','after'):
        blocks=[]
        for d in range(8):
            for split in ('T0','T1'):
                name=f'difficulty-D{d}-{split}'
                if name not in available:continue
                folder,_=available[name];bank=read(folder/'bank.json')
                with np.load(folder/f'matched-{phase}.npz') as f:features=f['features']
                boards=np.stack([encode(restore(levels[f'D{d}-{split}']),state_from(s)).ravel() for s in bank['states']])
                y=np.array([[int(x['push']),*map(int,x['solvable_actions'])] for x in bank['labels']])
                blocks.append(dict(model=name,split=split,features=features,boards=boards,y=y))
        tr=[b for b in blocks if b['split']=='T0'];te=[b for b in blocks if b['split']=='T1']
        if not tr or not te:continue
        for representation in ('features','boards'):
            X=np.concatenate([b[representation] for b in tr]);Y=np.concatenate([b['y'] for b in tr])
            for block in te:
                for target,label in enumerate(('push','solvable_up','solvable_down','solvable_left','solvable_right')):
                    pred=ridge_decode(X,Y[:,target],block[representation]);truth=block['y'][:,target]
                    learning.append(dict(phase=phase,model=block['model'],representation=representation,target=label,
                        correct=int(((pred>=.5)==truth).sum()),states=len(truth),
                        brier=float(np.mean((np.clip(pred,0,1)-truth)**2)),fit_families=[b['model'] for b in tr],
                        majority_baseline_accuracy=float(np.mean(truth==int(Y[:,target].mean()>=.5)))))
    annotation=SESSION/'neurons/annotations.parquet'
    if annotation.exists():
        n=pd.read_parquet(annotation)
        for col in ('type','superclass','class','subclass','somaNeuromere','nt'):
            if col not in n:continue
            values=n[col].fillna('unannotated').astype(str)
            for name,group in n.groupby(values,sort=False):
                anatomy.append(dict(annotation=col,label=name,cells=len(group),direct_inputs=int(group.direct_input.sum()),
                    observed=int(group.observed_by_policy.sum()),missing=name=='unannotated',
                    interpretation='soma location' if col=='somaNeuromere' else 'source annotation'))
    table(SESSION/'tables/cross-family-memory-decoding.csv',memory)
    table(SESSION/'tables/cross-family-learning-decoding.csv',learning)
    table(SESSION/'tables/annotation-coverage.csv',anatomy)
    return dict(memory_rows=len(memory),learning_rows=len(learning),annotation_rows=len(anatomy),
        fit='M0 trained histories for memory; T0 maps for learning; no held-out refitting')


def dense_group_analysis():
    """Stream lossless cell-by-time arrays in 1-ms chunks; no multi-GB RAM inflation."""
    import zipfile
    available={r.get('model'):(p,r) for p,r in completed_runs() if r.get('model')}
    generated=[]
    for line in ('learning','memory'):
        groupfile=SESSION/f'runs/discover-{line}/groups.json'
        if not groupfile.exists():continue
        groups=read(groupfile)['groups'];selections=[(g['id']+'-'+side,np.array(g[side])) for g in groups for side in ('indices','controls')]
        for name,(folder,result) in available.items():
            if (result.get('kind')=='static')!=(line=='learning'):continue
            for path in sorted((folder/'dense').glob('*.npz')):
                dest=SESSION/'group-traces'/name/path.name;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():generated.append(str(dest.relative_to(SESSION)));continue
                with zipfile.ZipFile(path) as archive,archive.open('values.npy') as stream:
                    version=np.lib.format.read_magic(stream)
                    shape,fortran,dtype=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(stream)
                    assert not fortran and len(shape)==3 and shape[1:]==(5,167106) and dtype==np.dtype('float32')
                    nbytes=int(np.prod(shape[1:]))*dtype.itemsize;chunks=[]
                    for t in range(shape[0]):
                        raw=stream.read(nbytes)
                        if len(raw)!=nbytes:raise ValueError('Truncated dense trajectory')
                        values=np.frombuffer(raw,dtype=dtype).reshape(shape[1:])
                        chunks.append(np.stack([values[:,ids].mean(1) for _,ids in selections],1))
                with np.load(path) as f:milliseconds=f['milliseconds']
                save_npz(dest,values=np.stack(chunks),milliseconds=milliseconds,fields=np.array(FIELDS),groups=np.array([label for label,_ in selections]))
                generated.append(str(dest.relative_to(SESSION)))
    write(SESSION/'tables/dense-group-trace-index.json',generated)
    return len(generated)


def analyze():
    for name in ('tables','figures','atlas'):(SESSION/name).mkdir(parents=True,exist_ok=True)
    if not (SESSION/'plan.json').exists():raise RuntimeError('Run mechanisms prepare first')
    existing=existing_data();params=parameter_analysis();static=static_analysis();memory=memory_analysis();causal=intervention_analysis()
    cross_family=cross_family_analysis();dense_traces=dense_group_analysis()
    current=read(SESSION/'latest-status.json') if (SESSION/'latest-status.json').exists() else {}
    jobs=read(SESSION/'plan.json')['jobs'];complete=list(completed_runs())
    summary=dict(generated_unix=time.time(),existing=existing,parameters=params,static=static,memory=memory,causal=causal,
        cross_family=cross_family,dense_group_traces=dense_traces,
        completed_jobs=len(complete),planned_jobs=len(jobs),training_updates=0,
        all_complete=len(complete)==len(jobs),statistical_unit='map family; seed 10 only')
    write(SESSION/'analysis.json',summary)
    lines=['# Neural dynamical mechanisms of task acquisition and short-term memory','',
        'Research focus: how learning changes coordination between neural representations and behavioral readout in a fly-connectome-constrained model, and how historical information is stored, transformed, and used for decisions.','',
        f"Reconstructed existing runs: {existing['runs']}; learning events: {existing['events']}; independently solved difficulty tasks: {existing['difficulty_solved']}/{existing['difficulty']}. Frozen replay jobs retrieved: {len(complete)}/{len(jobs)}.",'',
        '## Scope supported by existing results','',
        'The 24 difficulty tasks support task acquisition on specific maps with one seed. Covarying box count, board size, and shortest paths are not independent causal factors; timings from different hardware are presented separately.',
        'M0 clearing, swapping, and restoration support a behavioral role for historical state; S1/S2 and untrained timings bound its scope. Readout-parameter changes during frozen-brain training are not continued learning by the brain itself.',
        'All existing stops, partial solutions, and implementation diagnostics remain in the run registry. Unsolved outcomes are not all model-capacity limits.','',
        '## Six analysis outputs','',
        f"1. Acquisition: run-index, learning-events, and training-controls; {existing['events']} events.",
        f"2. Brain and readout: {static['decomposition_rows']} four-combination rows; margins with a fixed competitor add exactly.",
        f"3. Whole-brain atlas: {static['type_summary_rows']} type-summary rows; cell tables use bodyId, while parameter changes use shared types.",
        f"4. Memory geometry: {memory['geometry_rows']} cue-difference/readout-subspace rows; cross-time decoders fit trained histories and evaluate held-out histories.",
        f"5. Causal interventions: {causal['causal_rows']} retrieved paired rows; sham and restoration must pass, with T1/S1 separate from discovery.",
        f"6. Mechanistic paths: {causal['path_rows']} signed-connection summary rows; paths propose mechanisms, while behavior requires independent intervention.",'',
        f"Cross-map interpretive validation: {cross_family['learning_rows']} learning rows and {cross_family['memory_rows']} memory rows; coordinates and decoders fit only T0 and M0, respectively. Extracted {dense_traces} candidate/matched-group traces at 1 ms resolution.",'',
        'Unretrieved modules remain pending analysis; placeholders or missing data are not negative results.','',
        '## Mathematical interpretation and evidence limits','',
        'For the actual normalized and clipped features z, center the four action weights to obtain Wc. Its row space has at most three dimensions; projected cue-difference energy describes directions sensitive to the current readout. Bias enters score decomposition. The null space denotes insensitivity of current action differences, not irrelevance to later dynamics.',
        'Simultaneous current/voltage summaries at 1 ms resolution support dynamics and decoding. The actual policy combines current at window time 14 ms with voltage at the window end; these are labeled separately. PCA, decoders, and low-dimensional state models do not generate actions.',
        'Low-dimensional models split complete histories. Failed held-out predictions remain reported; stable eigenvalues or similar trajectories alone do not establish an attractor. Cells, time points, prefixes, and delay combinations do not add independent training replicates.','',
        '## Biological evidence table','',
        '| Model finding/question | Supporting data | Alternatives and assumptions | Testable prediction |',
        '|---|---|---|---|',
        '| Task acquisition and readout coordination | Original-start solutions and four-combination decomposition | Artificial sensory/action mapping and map-specific policy | Altering representation–readout coordination should selectively change relevant action margins |',
        '| Cue state participates in decisions | M0 clearing, swapping, restoration, and new component interventions | Clearing can disrupt general computational state | Hidden-goal impairment should exceed matched visible-goal impairment, with restoration returning to baseline |',
        '| Adaptation or synaptic resources maintaining history | Awaiting resource/adaptation interventions and trajectories | 200/300 ms are model constants, not measured biological time constants | Candidate-component and holding-window effects should show timing specificity |',
        '| History encoding converted into action | Awaiting readout geometry and S2 context analysis | Decodable information may not be used by the policy | The same cue should enter different action directions after context transformation |',
        '| Distributed group propagation | Awaiting candidate/matched group, window, and dose results | Type, degree, activity, and artificial observation-set confounds | Independent-layout confirmation with weaker matched-group effects is required for group specificity |','',
        '## Method references','',
        '- [Mechanistic hypotheses from connectome-constrained models](https://www.nature.com/articles/s41586-024-07939-3).',
        '- [dPCA](https://arxiv.org/abs/1410.6031); [dynamic coding and stable readout](https://johndmurray.org/papers/murray_2017_pnas.pdf).',
        'This study concerns simulated neurons and computational mechanisms, not direct evidence of Sokoban ability or memory mechanisms in real flies.','']
    (SESSION/'analysis.en.md').write_text('\n'.join(lines),encoding='utf-8')
    return {k:summary[k] for k in ('completed_jobs','planned_jobs','training_updates','all_complete')}
