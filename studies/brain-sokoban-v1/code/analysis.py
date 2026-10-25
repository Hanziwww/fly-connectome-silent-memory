"""Map-family summaries, paired effects, activity analyses and Chinese interpretation."""
import csv
import json
from pathlib import Path
import numpy as np
from registry import STUDY,read,write


def rows(path):
    return [json.loads(line) for line in path.open(encoding='utf-8')] if path.exists() else []


def table(path,data):
    if not data:return
    columns=list(dict.fromkeys(k for r in data for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(data)


def ridge(train_x,train_y,test_x,penalty=10.):
    mean=train_x.mean(0);scale=np.maximum(train_x.std(0),1e-5)
    x=(train_x-mean)/scale;test=(test_x-mean)/scale
    center=train_y.mean(0)
    # Dual formulation is cheaper when discovery states are fewer than features.
    if x.shape[0]<x.shape[1]:
        coef=x.T@np.linalg.solve(x@x.T+penalty*np.eye(len(x)),train_y-center)
    else:coef=np.linalg.solve(x.T@x+penalty*np.eye(x.shape[1]),x.T@(train_y-center))
    return test@coef+center


def decode(features,boards,bank):
    from itertools import permutations
    from registry import read
    levels={l['id']:l for l in read(STUDY/'tasks/levels.json')}
    train=np.array([r['role']=='discovery' for r in bank]);test=np.array([r['role']=='confirmation' for r in bank])
    if not train.any() or not test.any():return []
    families=np.array([r['family'] for r in bank]);player=np.array([r['state'][0] for r in bank])
    action=np.array([r['action'] for r in bank]);push=np.array([r['push'] for r in bank],dtype=int)
    remaining=np.array([r['remaining'] for r in bank],float)
    distance=[];composition=[];phase=[]
    for r in bank:
        l=levels[r['task']];boxes=r['state'][1];goals=l['goals']
        distance.append(min(sum(abs(b//10-g//10)+abs(b%10-g%10) for b,g in zip(boxes,order)) for order in permutations(goals)))
        composition.append([l['boxes'],l['floor_cells'],len(l.get('articulation_cells',[]))])
        phase.append(r['index']/max(l['optimal_steps'],1))
    base=np.column_stack([np.eye(100)[player],np.asarray(composition),phase])
    sources=dict(brain=features,board=boards.reshape(len(boards),-1),nuisance=base,
                 brain_plus_nuisance=np.column_stack([features,base]))
    result=[]
    for target,y,classification in [('witness_action',action,True),('push',push,True),
                                     ('remaining_moves',remaining,False),('box_goal_manhattan',np.array(distance),False)]:
        nuisance=base if target=='witness_action' else np.column_stack([base,np.eye(4)[action]])
        sources['nuisance']=nuisance;sources['brain_plus_nuisance']=np.column_stack([features,nuisance])
        yt=np.eye(4 if target=='witness_action' else 2)[y] if classification else y[:,None]
        for name,x in sources.items():
            pred=ridge(np.asarray(x[train],float),yt[train],np.asarray(x[test],float))
            values=(pred.argmax(1)==y[test]).astype(float) if classification else np.abs(pred[:,0]-y[test])
            for family in sorted(set(families[test])):
                sel=families[test]==family
                result.append(dict(target=target,source=name,family=family,states=int(sel.sum()),
                    metric='accuracy' if classification else 'MAE',value=float(values[sel].mean()),
                    decoder_penalty=10.,split='T0 discovery -> T1 confirmation'))
    return result


def activity_analysis(folder,out,plt):
    import sys
    sys.path.insert(0,str(STUDY.parents[1]/'sokoban'))
    import bootstrap
    from game import encode,restore,state_from
    bank=read(folder/'bank.json');levels={l['id']:l for l in read(STUDY/'tasks/levels.json')}
    boards=np.stack([encode(restore(levels[r['task']]),state_from(r['state'])) for r in bank])
    names=[n for n in ('before','mid','after') if (folder/(n+'-features.npz')).exists()]
    data={n:np.load(folder/(n+'-features.npz'))['features'].astype(float) for n in names}
    pooled=np.concatenate(list(data.values()));mean=pooled.mean(0);scale=np.maximum(pooled.std(0),1e-4)
    _,singular,vt=np.linalg.svd((pooled-mean)/scale,full_matrices=False);basis=vt[:3].T
    explained=(singular[:3]**2/(singular**2).sum()).tolist()
    fig,ax=plt.subplots(figsize=(6,4))
    for name,x in data.items():
        pc=((x-mean)/scale)@basis
        ax.scatter(pc[:,0],pc[:,1],s=8,alpha=.35,label=name)
    ax.set(xlabel='Shared PC1',ylabel='Shared PC2',title=folder.name);ax.legend();fig.tight_layout()
    fig.savefig(out/(folder.name+'-pca.png'),dpi=180);plt.close(fig)
    decoded=[]
    for phase,x in data.items():decoded.extend(dict(phase=phase,**r) for r in decode(x,boards,bank))
    table(out/(folder.name+'-decoding.csv'),decoded)
    # State-wise correlation geometry, compared in a shared matched-state order.
    similarities={}
    upper=np.triu_indices(len(bank),1)
    for name,x in data.items():
        z=x-x.mean(1,keepdims=True);z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-9)
        similarities[name]=(z@z.T)[upper]
    rsa={f'{a}__{b}':float(np.corrcoef(similarities[a],similarities[b])[0,1]) for i,a in enumerate(names) for b in names[i+1:]}
    instantaneous=np.load(folder/'after-features.npz')['instantaneous'].astype(float)
    train=np.array([r['role']=='discovery' for r in bank]);test=np.array([r['role']=='confirmation' for r in bank])
    times=[0,6,13,19,29,39];generalization=np.zeros((len(times),len(times)))
    # Dimensionality reduction fits discovery histories only; no confirmation leakage.
    tr=instantaneous[train][:,times].reshape(-1,512);mu=tr.mean(0);sd=np.maximum(tr.std(0),1e-4)
    _,_,v=np.linalg.svd((tr-mu)/sd,full_matrices=False);projection=v[:20].T
    activity=((instantaneous-mu)/sd)@projection
    y=np.eye(4)[np.array([r['action'] for r in bank])];family=np.array([r['family'] for r in bank])
    for i,t in enumerate(times):
        for j,u in enumerate(times):
            pred=ridge(activity[train,t],y[train],activity[test,u])
            acc=(pred.argmax(1)==y[test].argmax(1)).astype(float)
            generalization[i,j]=np.mean([acc[family[test]==f].mean() for f in set(family[test])])
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(generalization,vmin=0,vmax=1,cmap='viridis')
    ax.set(xticks=range(6),yticks=range(6),xticklabels=np.array(times)+1,yticklabels=np.array(times)+1,
           xlabel='Test time (ms)',ylabel='Train time (ms)',title='Cross-family action decoding')
    fig.colorbar(im,ax=ax,label='Mean family accuracy');fig.tight_layout();fig.savefig(out/(folder.name+'-temporal.png'),dpi=180);plt.close(fig)
    np.savez_compressed(out/(folder.name+'-representations.npz'),pca_basis=basis,mean=mean,scale=scale,
        temporal_generalization=generalization,times_ms=np.array(times)+1)
    sensitivity=None
    if (folder/'precision-sensitivity.npz').exists():
        alt=np.load(folder/'precision-sensitivity.npz')['features']
        sensitivity=dict(max_abs_feature_change=float(np.abs(data['after']-alt).max()),
            mean_abs_feature_change=float(np.abs(data['after']-alt).mean()),
            decoded=decode(alt,boards,bank))
    write(out/(folder.name+'-mechanism.json'),dict(shared_pca_explained_variance=explained,rsa=rsa,
        precision_sensitivity=sensitivity,units='map families; state counts describe coverage'))


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'savefig.bbox':'tight','font.size':9})
    out=STUDY/'analysis';out.mkdir(exist_ok=True)
    schedule=read(STUDY/'schedule.json');levels=read(STUDY/'tasks/levels.json');lookup={l['id']:l for l in levels}
    records=[];results={}
    for folder in sorted((STUDY/'runs').glob('*')):
        if not (folder/'result.json').exists():continue
        r=read(folder/'result.json');results[folder.name]=r;c=r.get('config',{})
        task=c.get('task',r.get('task'));l=lookup.get(task,{})
        v=read(folder/'verification.json') if (folder/'verification.json').exists() else {}
        records.append(dict(run=folder.name,module=c.get('module'),task=task,family=c.get('family_id',l.get('family_id')),
            tier=l.get('tier'),condition=c.get('condition',{}).get('name') if isinstance(c.get('condition'),dict) else c.get('condition'),
            seed=c.get('seed'),precision=c.get('precision'),batch=c.get('batch'),parent=c.get('parent'),
            status=r['status'],updates=r.get('updates'),training_seconds=r.get('training_seconds'),wall_seconds=r.get('wall_seconds'),
            solved=r.get('final_solved'),verified_solved=v.get('original_solved'),verification=v.get('status'),
            final_steps=r.get('final_steps'),peak_vram_gib=r.get('resources',{}).get('peak_allocated_gib'),
            code_version=c.get('code_version')))
    done={r['run'] for r in records}
    for j in schedule:
        if j['id'] not in done:
            progress=STUDY/'runs'/j['id']/'progress.json'
            active=read(progress) if progress.exists() else {}
            records.append(dict(run=j['id'],module=j['module'],task=j['task'],
                status='running' if active.get('status')=='training' else 'pending',updates=active.get('updates'),
                training_seconds=active.get('elapsed')))
    table(out/'run-registry.csv',records)
    table(out/'tasks.csv',[{k:l.get(k) for k in ('id','family_id','split','tier','size','boxes','optimal_steps','floor_cells',
        'witness_pushes','shortest_path_count_capped','floor_branch_cells')} for l in levels])
    # Geometric task atlas, emitted before training outcomes are available.
    fig,axes=plt.subplots(8,4,figsize=(10,18))
    for row in range(8):
        for col,suffix in enumerate(('T0','T1','T2','H')):
            l=lookup[f'D{row}-{suffix}'];ax=axes[row,col];img=np.zeros(100);img[l['walls']]=1
            ax.imshow(img.reshape(10,10),cmap='Greys',vmin=0,vmax=1)
            ax.scatter(np.array(l['goals'])%10,np.array(l['goals'])//10,facecolors='none',edgecolors='#dfab44',s=85,lw=1.8)
            for cells,color,marker in [(l['start'][1],'#469fa8','s'),([l['start'][0]],'#d56860','o')]:
                ax.scatter(np.array(cells)%10,np.array(cells)//10,c=color,marker=marker,s=22)
            ax.set_title(f"{l['id']} | {l['optimal_steps']} moves");ax.set_xticks([]);ax.set_yticks([])
    fig.tight_layout();fig.savefig(out/'task-atlas.png',dpi=150);plt.close(fig)
    accepted_stops={'learned_solved','already_solved','stagnation','segment_limit','frozen_baseline'}
    difficulty=[r for r in records if r.get('module')=='difficulty' and r['status'] in accepted_stops and r.get('verification')=='verified']
    if difficulty:
        fig,axes=plt.subplots(1,2,figsize=(10,3.5))
        for r in difficulty:
            marker='o' if r.get('verified_solved') else 'x';color='#238c78' if r.get('verified_solved') else '#ba5a55'
            axes[0].scatter(r['tier'],r.get('updates') or 0,marker=marker,color=color)
            axes[1].scatter(r['tier'],(r.get('training_seconds') or 0)/60,marker=marker,color=color)
        for ax in axes:ax.set(xlabel='Difficulty tier',xticks=range(8))
        axes[0].set_ylabel('Optimizer updates');axes[1].set_ylabel('Training minutes')
        fig.tight_layout();fig.savefig(out/'difficulty-learning.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(8,3,figsize=(11,18),squeeze=False)
        for d in range(8):
            for i in range(3):
                task=f'D{d}-T{i}';h=rows(STUDY/'runs'/('difficulty-'+task)/'history.jsonl');ax=axes[d,i]
                if h:ax.plot([r['elapsed']/60 for r in h],[r['depth'] for r in h],color='#238c78')
                ax.set(title=task,xlabel='Training minutes',ylabel='Suffix depth')
        fig.tight_layout();fig.savefig(out/'curriculum-learning-curves.png',dpi=180);plt.close(fig)
    if (STUDY/'benchmark.json').exists():
        bench=read(STUDY/'benchmark.json')
        if bench.get('candidates'):
            table(out/'speed-comparison.csv',bench['candidates'])
            fig,ax=plt.subplots(figsize=(6,3));cs=bench['candidates']
            bars=ax.bar([c['id'] for c in cs],[c['total_training_seconds'] for c in cs],color=['#238c78' if c['eligible'] else '#999999' for c in cs])
            for bar,c in zip(bars,cs):
                if not c['eligible']:bar.set_hatch('//')
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+12,'Both solved' if c['eligible'] else 'Failed early',ha='center',fontsize=8)
            ax.set_ylabel('Observed training seconds');ax.set_ylim(0,max(c['total_training_seconds'] for c in cs)*1.12)
            fig.tight_layout();fig.savefig(out/'speed-comparison.png',dpi=180);plt.close(fig)
    # Every comparison pairs exactly the same map and inherited checkpoint.
    contrasts=[]
    for r in records:
        if r.get('module')!='controls' or r['status'] not in accepted_stops or r.get('verification')!='verified':continue
        ref=next((d for d in difficulty if d['task']==r['task']),None)
        if ref:
            contrasts.append(dict(task=r['task'],family=r['family'],condition=r['condition'],
                success_difference=int(bool(r.get('verified_solved')))-int(bool(ref.get('verified_solved'))),
                update_difference=(r.get('updates') or 0)-(ref.get('updates') or 0),
                seconds_difference=(r.get('training_seconds') or 0)-(ref.get('training_seconds') or 0),
                condition_status=r['status'],reference_status=ref['status']))
    table(out/'paired-training-effects.csv',contrasts)
    if contrasts:
        arms=sorted({r['condition'] for r in contrasts});fig,ax=plt.subplots(figsize=(9,4))
        for i,arm in enumerate(arms):
            rs=[r for r in contrasts if r['condition']==arm]
            ax.scatter([i]*len(rs),[r['seconds_difference']/60 for r in rs],s=25,color='#326d8e')
        ax.axhline(0,color='#999999',lw=1);ax.set(xticks=range(len(arms)),xticklabels=arms,ylabel='Paired training minutes vs joint')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right');fig.tight_layout();fig.savefig(out/'training-method-effects.png',dpi=180);plt.close(fig)
    generalization=[];retention=[];restorations=[]
    for p in (STUDY/'runs').glob('*/generalization.json'):
        g=read(p)
        for r in g['records']:
            generalization.append(dict(run=g['run_id'],task=r['task'],family=r['family'],status=r['status'],steps=r['steps'],
                split=r['split'],start_used_for_training=r.get('start_used_for_training'),start_visited=r.get('start_visited_in_training_assessment')))
        for r in g['retention']:retention.append(dict(run=g['run_id'],task=r['task'],status=r['status'],steps=r['steps']))
        for name,r in g['restoration'].items():restorations.append(dict(run=g['run_id'],pairing=name,task=r['task'],status=r['status']))
    table(out/'generalization.csv',generalization);table(out/'retention.csv',retention);table(out/'parameter-restoration.csv',restorations)
    if retention:
        matrix=np.full((8,8),np.nan)
        base=results.get('difficulty-D0-T0',{})
        if base:matrix[0,0]=float(bool(base.get('final_solved')))
        for r in retention:
            if r['run'].startswith('curriculum-D'):
                matrix[int(r['run'].split('D')[-1]),int(r['task'][1])]=float(r['status']=='solved')
        fig,ax=plt.subplots(figsize=(5,4));ax.imshow(matrix,vmin=0,vmax=1,cmap='RdYlGn')
        ax.set(xlabel='Previously trained task tier',ylabel='Current curriculum stage',xticks=range(8),yticks=range(8));fig.tight_layout()
        fig.savefig(out/'retention-matrix.png',dpi=180);plt.close(fig)
    relearning=[]
    for j in schedule:
        if j['module']!='parameter_memory' or not j['id'].endswith('-A2'):continue
        a1=results.get(j['restore_from']);b=results.get(j['parent']);a2=results.get(j['id'])
        if not (a1 and b and a2):continue
        relearning.append(dict(pair=j['id'],A_acquired=a1.get('final_solved'),B_acquired=b.get('final_solved'),
            A_reacquired=a2.get('final_solved'),first_updates=a1.get('updates'),relearning_updates=a2.get('updates'),
            updates_saved=(a1.get('updates') or 0)-(a2.get('updates') or 0),
            first_seconds=a1.get('training_seconds'),relearning_seconds=a2.get('training_seconds')))
    table(out/'relearning-savings.csv',relearning)
    if relearning:
        fig,ax=plt.subplots(figsize=(7,4));x=np.arange(len(relearning))
        ax.bar(x-.18,[r['first_updates'] or 0 for r in relearning],.36,label='First learning')
        ax.bar(x+.18,[r['relearning_updates'] or 0 for r in relearning],.36,label='Relearning')
        ax.set(xticks=x,xticklabels=[r['pair'].replace('memory-','').replace('-A2','') for r in relearning],ylabel='Updates');ax.legend()
        fig.tight_layout();fig.savefig(out/'relearning-savings.png',dpi=180);plt.close(fig)
    memory=[];lesions=[];cue_decoding=[]
    for p in (STUDY/'runs').glob('dynamic-*/verification.json'):
        value=read(p)
        if 'results' not in value:continue
        c=read(p.parent/'config.json')
        for r in value['results']['probes']:
            memory.append(dict(task=c['task'],condition=c['condition'],delay_ms=r['delay_ms'],prefix_windows=r['prefix_windows'],
                history_split=r['history_split'],accuracy=r['correct']/r['total']))
        for r in value['interventions']:lesions.append(dict(task=c['task'],condition=c['condition'],delay_ms=r['delay_ms'],
            prefix_windows=r['prefix_windows'],intervention=r['intervention'],accuracy=r['correct']/r['total']))
        ps=value['results']['probes'];observations=[(r['features'],r['cue'],p['history_split'],p['delay_ms']) for p in ps for r in p['records'] if r.get('features') is not None]
        if observations:
            x=np.asarray([o[0] for o in observations]);y=np.eye(2)[[o[1] for o in observations]]
            train=np.array([o[2]=='train' for o in observations]);test=~train
            predicted=ridge(x[train],y[train],x[test]);accuracy=(predicted.argmax(1)==y[test].argmax(1))
            for d in sorted({o[3] for o in observations if o[2]!='train'}):
                sel=np.array([o[3]==d for o in observations])[test]
                cue_decoding.append(dict(task=c['task'],condition=c['condition'],delay_ms=d,
                    heldout_history_accuracy=float(accuracy[sel].mean()),identical_current_board_accuracy=.5))
    table(out/'memory-delays.csv',memory);table(out/'memory-state-interventions.csv',lesions)
    table(out/'hidden-cue-decoding.csv',cue_decoding)
    if memory:
        fig,ax=plt.subplots(figsize=(7,4))
        for kind in sorted({r['condition'] for r in memory}):
            ds=sorted({r['delay_ms'] for r in memory});means=[]
            for d in ds:
                by_map=[np.mean([r['accuracy'] for r in memory if r['condition']==kind and r['delay_ms']==d and r['task']==m])
                    for m in sorted({r['task'] for r in memory if r['condition']==kind})]
                means.append(np.mean(by_map))
            ax.plot(ds,means,'o-',label=kind)
        ax.set(xlabel='Blank delay (ms)',ylabel='Mean layout accuracy',ylim=(-.02,1.02));ax.legend();fig.tight_layout()
        fig.savefig(out/'memory-delay-curves.png',dpi=180);plt.close(fig)
    mechanism_count=0
    for p in (STUDY/'activity').glob('*/recording.json') if (STUDY/'activity').exists() else []:
        if read(p).get('representative'):
            activity_analysis(p.parent,out,plt);mechanism_count+=1
    causal=[]
    for p in (STUDY/'activity').glob('*/interventions.json') if (STUDY/'activity').exists() else []:
        x=read(p);base={r['task']:r['status']=='solved' for r in x['baseline']['records']}
        for trial in x['records']:
            for family in sorted({r['family'] for r in trial['records']}):
                rs=[r for r in trial['records'] if r['family']==family]
                causal.append(dict(run=p.parent.name,group=trial['group'],family=family,matched_control=trial['matched_control'],
                    strength=trial['strength'],window=str(trial['window_ms']),
                    paired_success_change=float(np.mean([int(r['status']=='solved')-int(base[r['task']]) for r in rs]))))
    table(out/'causal-family-effects.csv',causal)
    if causal:
        fig,ax=plt.subplots(figsize=(8,4));keys=sorted({(r['strength'],r['window'],r['matched_control']) for r in causal})
        for i,key in enumerate(keys):
            values=[r['paired_success_change'] for r in causal if (r['strength'],r['window'],r['matched_control'])==key]
            ax.scatter(np.full(len(values),i),values,s=12,alpha=.4)
            ax.plot([i-.2,i+.2],[np.mean(values)]*2,color='black',lw=2)
        ax.set(xticks=range(len(keys)),xticklabels=[f'{s}/{w}/'+('control' if c else 'target') for s,w,c in keys],ylabel='Paired success change')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right');fig.tight_layout();fig.savefig(out/'causal-interventions.png',dpi=180);plt.close(fig)
    successes=sum(bool(r.get('verified_solved')) for r in difficulty)
    acquired=sum(r['status']=='learned_solved' and bool(r.get('verified_solved')) for r in difficulty)
    pending=sum(r['status'] in ('pending','running') for r in records)
    summary=dict(completed_runs=sum(r['status'] not in ('pending','running') for r in records),scheduled_pending=pending,
        difficulty_evaluated=len(difficulty),difficulty_verified_solved=successes,difficulty_new_acquisitions=acquired,
        mechanism_models=mechanism_count,memory_layout_conditions=len({(r['task'],r['condition']) for r in memory}),
        overall_complete=pending==0 and len(difficulty)==24 and mechanism_count==4 and len({(r['task'],r['condition']) for r in memory})==15)
    write(STUDY/'summary.json',summary)
    paragraphs=[
        '# Fly-connectome Sokoban experiments: data and interpretation',
        f"Recorded runs: {summary['completed_runs']} (including implementation diagnostics); scheduled items remaining: {pending}.",
        f"Difficulty maps completed: {len(difficulty)}/24; independent reload solutions: {successes}, including {acquired} newly acquired tasks. Success means autonomous completion, without a path-length gate.",
        'The fixed library contains 24 training maps, 8 untrained layouts, and 2 speed-calibration maps. All have bidirectional solving certificates; shortest paths, branching, and deadlock metrics describe structure only.',
        'Training controls are paired by map and parent checkpoint. Training and independent-reload success are separate; already solved maps are existing ability. Stagnation, computational-stage limits, implementation errors, numerical errors, and interruptions remain distinct.',
        'Hidden goals appear only during the cue. Both histories are tested at the same state, with a wrong first push solver-certified as irreversible. Common prefixes are controlled memory probes; subsequent full Sokoban behavior is autonomous.',
        'Activity analyses separate matched states from free trajectories. Decoders fit discovery families and evaluate independent confirmation families. Action decodability alone does not establish necessity, which requires group-output suppression, matched controls, and restoration.',
        f'Representative models analyzed: {mechanism_count}/4; memory verification covers {summary["memory_layout_conditions"]}/15 layout/model combinations. Missing results remain untested; no speculative values are filled in.',
        'Maps and map families are statistical units. Neurons, time points, actions, and checkpoints describe processes without increasing independent training replication. Training uses seed 10; conclusions are limited to this initialization lineage and tested maps.',
        'Files: run-registry.csv records all runs; paired-training-effects.csv stores paired controls; generalization.csv stores zero-update tests; memory-delays.csv and mechanism files follow completion of their experiments.',
    ]
    if difficulty:
        details=[]
        for tier in range(8):
            rs=[r for r in difficulty if r['tier']==tier]
            if not rs:continue
            wins=sum(bool(r.get('verified_solved')) for r in rs)
            times=[r['training_seconds']/60 for r in rs if r.get('training_seconds') is not None]
            details.append(f"D{tier}: {wins}/{len(rs)} reload-verified solutions, median training time {np.median(times):.2f} minutes" if times else f'D{tier}: {wins}/{len(rs)} reload-verified solutions')
        paragraphs.insert(3,'; '.join(details)+'. These describe a fixed task set; longer solutions do not by themselves imply harder neural training.')
        failed=[r for r in difficulty if not r.get('verified_solved')]
        if failed:
            counts={s:sum(r['status']==s for r in failed) for s in sorted({r['status'] for r in failed})}
            paragraphs.insert(4,'Stop reasons for unsolved tasks: '+json.dumps(counts,ensure_ascii=False)+'. Stagnation bounds performance under this configuration and finite continuation, not connectome-model capacity.')
    if contrasts:
        effects=[]
        for arm in sorted({r['condition'] for r in contrasts}):
            rs=[r for r in contrasts if r['condition']==arm]
            effects.append(f"{arm}: {len(rs)} paired maps, solution-rate difference {np.mean([r['success_difference'] for r in rs]):+.2f}, median time difference {np.median([r['seconds_difference'] for r in rs])/60:+.2f} minutes")
        paragraphs.append('Training controls: '+'; '.join(effects)+'. Time depends on success and stopping rules and must be read with solution outcomes. Structural and neutral-initialization controls change the starting point and alone cannot establish redundancy of a component in an existing P05 pairing.')
    if relearning:
        paragraphs.append('Parameter memory: '+'; '.join(f"{r['pair']} initial/relearning updates {r['first_updates']}/{r['relearning_updates']}, A initially acquired={r['A_acquired']}, reacquired={r['A_reacquired']}" for r in relearning)+
            '. Forgetting requires A to have been acquired before interference and subsequently lost; relearning savings require restored A solutions. Component-restoration margins are paired on the same A states in parameter-memory-margins.json.')
    if memory:
        observations=[]
        for kind in sorted({r['condition'] for r in memory}):
            trained=[r['accuracy'] for r in memory if r['condition']==kind and r['history_split']=='train']
            unseen=[r['accuracy'] for r in memory if r['condition']==kind and r['history_split']=='heldout']
            observations.append(f"{kind}: mean choice accuracy for trained/untrained histories {np.mean(trained):.3f}/{np.mean(unseen):.3f}")
        paragraphs.append('Dynamic memory: '+'; '.join(observations)+'. In a balanced two-choice task with identical current observations, a history-free policy is bounded by 0.5. Improvement with carried state shows use of input history in this model; identifying its state carrier requires component clearing, swapping, and restoration. Short-probe accuracy and full Sokoban solution rates are distinct.')
    if lesions:
        effects=[]
        for kind in ('all','voltage','current','adaptation','delay','swap','restore'):
            rs=[r for r in lesions if r['condition']=='joint' and r['intervention']==kind]
            if rs:effects.append(f"choice accuracy after {kind}: {np.mean([r['accuracy'] for r in rs]):.3f}")
        paragraphs.append('Joint persistent-state interventions: '+'; '.join(effects)+'. Systematic choice changes after opposite-cue state swaps identify the information used more specifically than a drop after clearing alone; restoration checks for irreversible implementation side effects.')
    if causal:
        means=[]
        for control in (False,True):
            rs=[r['paired_success_change'] for r in causal if r['matched_control']==control]
            means.append(np.mean(rs))
        paragraphs.append(f'In the confirmation family, mean solution changes for target and matched groups were {means[0]:+.3f} and {means[1]:+.3f}. This is a within-model causal intervention on selected groups, windows, and tasks. Increased activity, PCA separation, or decoding accuracy alone is not causal evidence and does not establish the same mechanism in animals.')
    if (STUDY/'backend.json').exists():
        b=read(STUDY/'backend.json');paragraphs.insert(4,f"Speed calibration selected {b['precision']}, batch {b['batch']}, based on learning and independent solutions on both calibration tasks, allowing different routes. Same-configuration FP32 recomputation tolerance is 0.001; actual errors are in gradient records. Activity conclusions require separate precision-sensitivity analysis.")
        if (STUDY/'benchmark.json').exists():
            candidates=read(STUDY/'benchmark.json').get('candidates',[])
            reference=next((c for c in candidates if c['id']=='fp64-b16' and c['eligible']),None)
            if reference and b['eligible']:
                paragraphs.append(f"Completed speed results: reference training {reference['total_training_seconds']:.2f} seconds, selected configuration {b['total_training_seconds']:.2f} seconds, ratio {reference['total_training_seconds']/b['total_training_seconds']:.2f}. This describes only the two certified calibration tasks; short runs ending on numerical error are not faster learning.")
    from analysis_extra import analyze_details
    detail=analyze_details(out,plt)
    if detail['hidden_goals']:
        ps=[]
        for kind in sorted({r['condition'] for r in detail['hidden_goals']}):
            rs=[r for r in detail['hidden_goals'] if r['condition']==kind]
            means=[np.mean([r['solved'] for r in rs if r['task']==m]) for m in sorted({r['task'] for r in rs})]
            ps.append(f'{kind}: full hidden-goal task solution rate {np.mean(means):.3f}')
        paragraphs.append('Autonomous behavior: '+'; '.join(ps)+'. Stratified short-probe and cross-step results are in memory-delays.csv and hidden-goal-behavior.csv; delays, cues, and decisions within a map are not independent map replicates.')
    paragraphs.append('Unanswered questions: unfinished runs support no mechanism conclusions; one seed does not estimate between-seed stability; fixed-task success does not establish arbitrary-layout solving. Model-state and intervention evidence is not directly equivalent to memory or thought in real flies.')
    (out/'analysis-report.en.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False));return 0
