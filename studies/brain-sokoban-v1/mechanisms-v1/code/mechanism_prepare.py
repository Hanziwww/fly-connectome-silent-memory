"""Lock the authorized frozen replay matrix; no experiment is trained here."""
import csv
import shutil
import time
from pathlib import Path
from mechanism_base import *


def sources():
    for p in sorted((STUDY / 'runs').glob('*/result.json')):
        yield 'local', p.parent
    for batch in ('remote-20261003', 'remote-supplement-20261003'):
        for p in sorted((STUDY / batch / 'results').glob('*/result.json')):
            yield batch, p.parent


def prepare():
    if (SESSION / 'plan.json').exists():
        plan = read(SESSION / 'plan.json')
        assert plan['scientific_sources'] == scientific_sources(), 'Scientific source drift'
        assert plan['code_manifest'] == code_manifest(), 'Mechanism code drift'
        return plan
    SESSION.mkdir(parents=True, exist_ok=True)
    registry = []; available = {}
    for origin, folder in sources():
        result = read_metadata(folder / 'result.json')
        config = read(folder / 'config.json') if (folder / 'config.json').exists() else {}
        verification = read_metadata(folder / 'verification.json') if (folder / 'verification.json').exists() else {}
        cond = config.get('condition', result.get('condition'))
        if isinstance(cond, dict): cond = cond.get('name')
        row = dict(id=folder.name, origin=origin, source=folder.relative_to(ROOT).as_posix(),
                   module=config.get('module'), task=config.get('task', result.get('task')),
                   condition=cond, status=result.get('status'), updates=result.get('updates'),
                   training_seconds=result.get('training_seconds'),
                   original_solved=verification.get('original_solved'),
                   solved=verification.get('solved'), total=verification.get('total'),
                   verification_status=verification.get('status'),
                   family=config.get('family_id', result.get('family')),
                   parameter_changes=result.get('parameter_changes', {}))
        rolls = verification.get('results', {}).get('rollouts', [])
        if rolls and row['solved'] is None:
            train = [r for r in rolls if r.get('delay_ms') in (0,40,160) and
                     r.get('prefix_windows') in (0,1,4) and not r.get('intervention')]
            row.update(solved=sum(r['status']=='solved' for r in train), total=len(train))
        registry.append(row); available[folder.name] = (folder, config, row)
    write(SESSION / 'registry.json', registry)
    models = []
    selections = [(f'difficulty-D{d}-T{t}', 'static', f'D{d}-T{t}') for d in range(8) for t in range(3)]
    selections += [(f'control-D{d}-{c}', 'static', f'D{d}-T0') for d in (1,3,5)
                   for c in ('brain_only', 'head_only', 'no_recovery')]
    selections += [('remote-D7-head_only', 'static', 'D7-T0')]
    selections += [(name, 'memory', task) for name, task in (
        ('remote-dynamic-M0-head_only', 'M0-audit'), ('remote-dynamic-M0-joint', 'M0-audit'),
        ('supp-M0-visibility', 'M0-visibility'), ('supp-S1-ring-brain', 'S1-ring'),
        ('supp-S2-context-brain', 'S2-context'))]
    for name, kind, task in selections:
        folder, cfg, row = available[name]
        bundle = SESSION / 'inputs/models' / name; bundle.mkdir(parents=True, exist_ok=True)
        checkpoints = {}
        for phase, filename in PHASES:
            if not (folder / filename).exists(): continue
            target = bundle / filename; shutil.copy2(folder / filename, target)
            checkpoints[phase] = dict(path=target.relative_to(SESSION).as_posix(), sha256=digest(target))
        assert 'before' in checkpoints and 'after' in checkpoints, name
        write(bundle / 'source-config.json', cfg)
        # Keep acceptance metadata compact; full original files remain untouched.
        write(bundle / 'acceptance.json', {k: row[k] for k in ('original_solved','solved','total','verification_status')})
        models.append(dict(id=name, kind=kind, task=task, config=cfg, source=row['source'],
            checkpoints=checkpoints, precision=cfg.get('precision','mixed'),
            representative=kind=='static' and name.startswith('difficulty-') and int(task[1]) in (1,3,5,7) and task.endswith(('T0','T1')),
            role=('discovery' if task.endswith('T0') else 'confirmation' if task.endswith('T1') else 'descriptive')
                if kind=='static' else 'discovery' if task.startswith('M0') else 'confirmation' if task=='S1-ring' else 'scope',
            source_acceptance={k:row[k] for k in ('original_solved','solved','total','verification_status')}))
    write(SESSION / 'models.json', models)
    inputs = SESSION / 'inputs'; inputs.mkdir(exist_ok=True)
    for source, name in ((STUDY/'tasks/levels.json','levels.json'),
                         (STUDY/'tasks/matched-bank.json','matched-bank.json'),
                         (STUDY/'remote-supplement-20261003/tasks.json','memory-tasks.json')):
        shutil.copy2(source, inputs/name)
    jobs = [dict(id='preflight', kind='preflight', dependencies=[])]
    for m in models:
        jobs.append(dict(id='record-'+m['id'],kind=m['kind'],model=m['id'],dependencies=['preflight']))
    learn_deps=['record-difficulty-D'+str(d)+'-T0' for d in range(8)]
    jobs += [dict(id='discover-learning',kind='discover_learning',dependencies=learn_deps),
             dict(id='discover-memory',kind='discover_memory',dependencies=['record-remote-dynamic-M0-head_only'])]
    for d in (1,3,5,7):
        name=f'difficulty-D{d}-T1'
        jobs.append(dict(id='intervene-'+name,kind='intervene_static',model=name,
                         dependencies=['discover-learning','record-'+name]))
    for name in ('remote-dynamic-M0-head_only','supp-M0-visibility','supp-S1-ring-brain','supp-S2-context-brain'):
        jobs.append(dict(id='components-'+name,kind='components',model=name,dependencies=['record-'+name]))
        jobs.append(dict(id='groups-'+name,kind='intervene_memory',model=name,
                         dependencies=['discover-memory','record-'+name]))
    jobs += [dict(id='pathways',kind='pathways',dependencies=['discover-learning','discover-memory'])]
    plan = dict(schema_version=1, created_unix=time.time(), seed=10, mode='frozen_replay_only',
        scientific_sources=scientific_sources(), code_manifest=code_manifest(),
        backend_data={name:digest(ROOT/name) for name in ('flybrain/cache/neurons.parquet','flybrain/cache/W_post_pre.npz',
            'sokoban/artifacts/mapping.npz','archive/experiments/campaign/run/neuron-mapping.npz')},
        protocol_sha256=digest(SESSION/'protocol.en.md'),
        inputs={p.relative_to(SESSION).as_posix():digest(p) for p in sorted(inputs.rglob('*')) if p.is_file()},
        gpu_lanes=[0,1,2,3], allocator_gib=11, minimum_free_gib=13, data_budget_gib=120,
        minimum_disk_free_gib=50, source_model_updates_allowed=False,
        tolerance=dict(atol=1e-5,rtol=1e-5), phases=['before','mid','after'],
        memory_delays=[0,20,40,80,160,320,640], memory_prefixes=[0,1,2,4,8],
        trained_delays=[0,40,160],trained_prefixes=[0,1,4],
        dense_delays=[160,640],dense_prefix=4,dense_decision_cap=8,action_cap=512,
        intervention_delays=[40,160,640],intervention_prefixes=[0,4],
        group_strengths=[.25,.5,1.],windows=['write','hold','read'],max_groups=3,
        minimum_noninput_cells=8, candidate_selection=dict(
            learning='mean absolute same-board voltage change across T0 maps; mean over eligible cells',
            memory='mean standardized cue contrast across trained M0 histories in g/v/adapt/res; noninput cells only'),
        matching=['cell_count','baseline_spike_probability','log_total_degree','observed_membership'],
        statistical_unit='map family; histories and cells do not increase independent model replication',
        narrative='Task acquisition, state memory and decision mechanisms; biological hypotheses explicitly separated',
        jobs=jobs)
    write(SESSION / 'plan.json', plan)
    write(SESSION / 'data-dictionary.json', dict(fields={
        'spikes':'binary event at 1 ms; summaries are per-bin event probabilities',
        'g':'simulator synaptic input current; model units', 'v':'membrane potential, mV',
        'adapt':'spike-triggered adaptation, mV', 'res':'presynaptic resource fraction',
        'features':'256 means of g at window +14 ms followed by 256 means of v-v_rest at window end',
        'instantaneous':'same-time pooled current and voltage; differs from policy feature sampling',
        'mid':'saved intermediate checkpoint; not assumed to equal half the wall time',
        'somaNeuromere':'source soma annotation, not synaptic neuropil membership'},
        pairing=['model','phase','task_family','input_history','cue','context','visibility','delay','prefix'],
        data_origin='simulated neural activity, not biological electrophysiology',
        discovery_confirmation=plan['candidate_selection']))
    return plan
