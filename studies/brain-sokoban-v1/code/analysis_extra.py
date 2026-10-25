"""Detailed behavior, resource and activity outputs from completed study artifacts."""
import numpy as np
from registry import STUDY,read


def analyze_details(out,plt):
    from analysis import rows,table
    speed=[];changes=[];failures=[]
    for p in (STUDY/'runs').glob('*/result.json'):
        r=read(p);c=r.get('config',{})
        for component,parameters in r.get('parameter_changes',{}).items():
            for name,value in parameters.items():
                changes.append(dict(run=p.parent.name,component=component,parameter=name,**value))
        if c.get('module')=='benchmark':
            gs=rows(p.parent/'gradients.jsonl');compute=r.get('training_compute',{})
            speed.append(dict(run=p.parent.name,task=c['task'],precision=c['precision'],batch=c['batch'],
                status=r['status'],training_seconds=r.get('training_seconds'),wall_seconds=r.get('wall_seconds'),
                updates=r.get('updates'),neuron_ms=compute.get('neuron_ms'),forward_calls=compute.get('forward_calls'),
                initialization_seconds=r.get('initialization_seconds'),
                peak_allocated_gib=r.get('resources',{}).get('peak_allocated_gib'),
                median_update_seconds=float(np.median([g['total_seconds'] for g in gs])) if gs else None,
                median_feature_forward_seconds=float(np.median([g['timings']['feature_forward'] for g in gs])) if gs else None,
                median_core_backward_seconds=float(np.median([g['timings']['core_recompute_backward'] for g in gs])) if gs else None))
        if (p.parent/'after.json').exists():
            b=read(p.parent/'after.json')
            if b['status']!='solved':
                failures.append(dict(run=p.parent.name,task=c.get('task'),training_stop=r['status'],
                    behavior_stop=b['status'],steps=b['steps'],last_state=b['decisions'][-1]['state'] if b['decisions'] else None,
                    suffix_depth=r.get('depth'),updates=r.get('updates')))
    table(out/'speed-resources-by-run.csv',speed);table(out/'parameter-changes.csv',changes)
    table(out/'failure-cases.csv',failures)
    interference=[]
    for folder in (STUDY/'runs').glob('memory-*'):
        for h in rows(folder/'history.jsonl'):
            margins={p['task']:p['mean_margin'] for p in h.get('probe_margins',[])}
            for p in h.get('probe_results',[]):
                interference.append(dict(run=folder.name,updates=h['updates'],elapsed=h['elapsed'],task=p['task'],
                    solved=p['status']=='solved',status=p['status'],mean_A_action_margin=margins.get(p['task'])))
    table(out/'interference-learning-curves.csv',interference)
    if interference:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for run,task in sorted({(r['run'],r['task']) for r in interference}):
            rs=[r for r in interference if r['run']==run and r['task']==task]
            axes[0].plot([r['updates'] for r in rs],[r['solved'] for r in rs],alpha=.7,label=run+':'+task)
            measured=[r for r in rs if r['mean_A_action_margin'] is not None]
            if measured:axes[1].plot([r['updates'] for r in measured],[r['mean_A_action_margin'] for r in measured],alpha=.7)
        axes[0].set(xlabel='Updates on current task',ylabel='Probe solved');axes[1].set(xlabel='Updates on current task',ylabel='Margin for prior A actions')
        axes[0].legend(fontsize=5);fig.tight_layout();fig.savefig(out/'interference-learning-curves.png',dpi=180);plt.close(fig)
    hidden=[];effects=[];probes=[]
    for p in (STUDY/'runs').glob('dynamic-*/verification.json'):
        v=read(p)
        if 'results' not in v:continue
        c=read(p.parent/'config.json');base={(r['delay_ms'],r['prefix_windows'],r['cue']):r for r in v['results']['rollouts']}
        bp={(r['delay_ms'],r['prefix_windows'],x['cue']):x for r in v['results']['probes'] for x in r['records']}
        for r in v['results']['rollouts']:
            hidden.append(dict(task=c['task'],condition=c['condition'],delay_ms=r['delay_ms'],prefix_windows=r['prefix_windows'],
                cue=r['cue'],history_split=r['history_split'],solved=r['status']=='solved',status=r['status'],steps=r['steps']))
        for r in v['intervention_rollouts']:
            before=base[(r['delay_ms'],r['prefix_windows'],r['cue'])]
            effects.append(dict(task=c['task'],condition=c['condition'],delay_ms=r['delay_ms'],prefix_windows=r['prefix_windows'],
                cue=r['cue'],intervention=r['intervention'],baseline_solved=before['status']=='solved',
                intervened_solved=r['status']=='solved',paired_success_change=int(r['status']=='solved')-int(before['status']=='solved'),
                status=r['status'],steps=r['steps']))
        for r in v['interventions']:
            for x in r['records']:
                before=bp[(r['delay_ms'],r['prefix_windows'],x['cue'])]
                probes.append(dict(task=c['task'],condition=c['condition'],delay_ms=r['delay_ms'],cue=x['cue'],
                    intervention=r['intervention'],baseline_choice=before['choice'],choice=x['choice'],
                    choice_changed=x['choice']!=before['choice'],accuracy_change=int(x['correct'])-int(before['correct']),
                    max_abs_logit_change=float(np.max(np.abs(np.asarray(x['logits'])-before['logits'])))))
    table(out/'hidden-goal-behavior.csv',hidden);table(out/'hidden-goal-intervention-effects.csv',effects)
    table(out/'state-swap-and-restoration.csv',probes)
    if hidden:
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,split in zip(axes,('train','heldout')):
            for kind in sorted({r['condition'] for r in hidden}):
                ds=sorted({r['delay_ms'] for r in hidden if r['history_split']==split});means=[]
                for d in ds:
                    rs=[r for r in hidden if r['condition']==kind and r['delay_ms']==d and r['history_split']==split]
                    by_map=[np.mean([s['solved'] for s in rs if s['task']==m]) for m in sorted({s['task'] for s in rs})]
                    means.append(np.mean(by_map) if by_map else np.nan)
                ax.plot(ds,means,'o-',label=kind)
            ax.set(title=split+' histories',xlabel='Delay (ms)',ylim=(-.02,1.02))
        axes[0].set_ylabel('Mean layout autonomous solve rate');axes[-1].legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/'hidden-goal-learning.png',dpi=180);plt.close(fig)
    descriptions=[]
    annotations=None
    if (STUDY/'activity/neuron-index.parquet').exists():
        import pandas as pd
        annotations=pd.read_parquet(STUDY/'activity/neuron-index.parquet')
    for p in (STUDY/'activity').glob('*/recording.json') if (STUDY/'activity').exists() else []:
        if not read(p).get('representative'):continue
        f=p.parent;stats={name:np.load(f/(name+'-neuron-stats.npz'))['mean'] for name in ('before','after')}
        if annotations is not None:
            grouped=[]
            for column in ('type','superclass','class','somaNeuromere'):
                if column not in annotations:continue
                labels=annotations[column].astype('string').fillna('unannotated')
                for phase,values in stats.items():
                    frame=pd.DataFrame(values.T,columns=['spikes','current','voltage','adaptation'])
                    frame['annotation']=labels.to_numpy();groups=frame.groupby('annotation',dropna=False)
                    means=groups.mean(numeric_only=True);counts=groups.size()
                    for label,row in means.iterrows():
                        grouped.append(dict(run=f.name,phase=phase,annotation_field=column,annotation=label,
                            neurons=int(counts[label]),**row.to_dict(),
                            interpretation='soma assignment, not synaptic neuropil membership' if column=='somaNeuromere' else 'source annotation'))
            table(out/(f.name+'-annotated-activity.csv'),grouped)
        fig,axes=plt.subplots(1,4,figsize=(12,3))
        for k,(ax,label) in enumerate(zip(axes,('Spike probability','Current','Membrane potential','Adaptation'))):
            for phase in ('before','after'):
                ax.hist(stats[phase][k],bins=60,histtype='step',density=True,label=phase)
            ax.set(xlabel=label,ylabel='Neuron density')
        axes[0].legend();fig.tight_layout();fig.savefig(out/(f.name+'-activity-atlas.png'),dpi=180);plt.close(fig)
        data=np.load(out/(f.name+'-representations.npz'))
        run=read(STUDY/'runs'/f.name/'after.json');decisions=run['decisions']
        if decisions:
            x=np.asarray([d['features'] for d in decisions]);pc=((x-data['mean'])/data['scale'])@data['pca_basis']
            fig,ax=plt.subplots(figsize=(5,4));ax.plot(pc[:,0],pc[:,1],lw=.8,c='#aaaaaa')
            dots=ax.scatter(pc[:,0],pc[:,1],c=np.arange(len(pc)),s=20,cmap='viridis')
            fig.colorbar(dots,ax=ax,label='Decision index');ax.set(xlabel='Shared PC1',ylabel='Shared PC2',title=run['status'])
            fig.tight_layout();fig.savefig(out/(f.name+'-free-trajectory.png'),dpi=180);plt.close(fig)
        for k,label in enumerate(('spikes','current','voltage','adaptation')):
            delta=stats['after'][k]-stats['before'][k]
            descriptions.append(dict(run=f.name,field=label,mean_change=float(delta.mean()),
                median_change=float(np.median(delta)),mean_abs_change=float(np.abs(delta).mean()),
                units='descriptive neuron distribution, not independent training repeats'))
    table(out/'activity-distribution-changes.csv',descriptions)
    return dict(hidden_goals=hidden,state_probes=probes,activity_changes=descriptions)
