"""Summarize saved memory outcomes and matched-history interventions; no inference."""
import argparse
import collections
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def counts(probes, rollouts):
    return dict(probe_correct=sum(p['correct'] for p in probes),
                probe_total=sum(p['total'] for p in probes),
                solved=sum(r['status']=='solved' for r in rollouts), total=len(rollouts),
                failure_types=dict(collections.Counter(r['status'] for r in rollouts)))


def summarize(root):
    records=[]
    for kind in ('carry_untrained','reset','head_only','joint','gru'):
        out=root/f'remote-dynamic-M0-{kind}'
        record=dict(condition=kind)
        if (out/'progress.json').exists(): record['progress']=load(out/'progress.json')
        if (out/'extensions.jsonl').exists():
            record['extensions']=[{k:v for k,v in json.loads(line).items() if k in
                ('phase','progress','min_choice_margin','phase_margin','solved_histories','required_histories')}
                for line in (out/'extensions.jsonl').read_text().splitlines() if line.strip()]
        if (out/'result.json').exists():
            result=load(out/'result.json')
            record.update({k:result[k] for k in ('status','updates','training_seconds','final_solved','resources') if k in result})
            record['parameter_changes']={component:{k:v['changed'] for k,v in values.items()}
                                         for component,values in result.get('parameter_changes',{}).items()}
        if (out/'verification.json').exists():
            verification=load(out/'verification.json')
            record['verification']={k:verification[k] for k in ('status','original_solved','pid','training_pid','model_sha256') if k in verification}
            final=verification.get('results',{})
            probes=final.get('probes',[]);rolls=final.get('rollouts',[])
            record['all_histories']=counts(probes,rolls)
            record['history_splits']={split:counts([p for p in probes if p['history_split']==split],
                                                   [r for r in rolls if r['history_split']==split])
                                     for split in ('train','heldout')}
            record['by_delay']={str(delay):counts([p for p in probes if p['delay_ms']==delay],
                                                 [r for r in rolls if r['delay_ms']==delay])
                                for delay in (0,20,40,80,160,320,640)}
            record['short_probes']=counts([p for p in probes if p['prefix_windows']==0],
                                         [r for r in rolls if r['prefix_windows']==0])
            record['matched_prefix4']=counts([p for p in probes if p['prefix_windows']==4],
                                             [r for r in rolls if r['prefix_windows']==4])
            record['interventions']={}
            for intervention in ('voltage','current','adaptation','delay','all','swap','restore'):
                ps=[p for p in verification.get('interventions',[]) if p['intervention']==intervention]
                rs=[r for r in verification.get('intervention_rollouts',[]) if r['intervention']==intervention]
                if not ps:continue
                value=counts(ps,rs)
                value['trained_delays']=counts([p for p in ps if p['delay_ms'] in (0,40,160)],
                                              [r for r in rs if r['delay_ms'] in (0,40,160)])
                if intervention=='swap':
                    value['opposite_history_choice_correct']=sum(
                        r['choice']==[3,2][1-r['cue']] for p in ps for r in p['records'])
                    baseline={(p['delay_ms'],r['cue']):r for p in probes if p['prefix_windows']==4 for r in p['records']}
                    value['partner_choice_matches']=sum(r['choice']==baseline[p['delay_ms'],1-r['cue']]['choice'] for p in ps for r in p['records'])
                    value['partner_logits_max_abs_difference']=max(abs(a-b) for p in ps for r in p['records']
                        for a,b in zip(r['logits'],baseline[p['delay_ms'],1-r['cue']]['logits']))
                if intervention=='restore':
                    baseline={(p['delay_ms'],r['cue']):r for p in probes if p['prefix_windows']==4 for r in p['records']}
                    value['original_choice_matches']=sum(r['choice']==baseline[p['delay_ms'],r['cue']]['choice'] for p in ps for r in p['records'])
                    value['original_logits_max_abs_difference']=max(abs(a-b) for p in ps for r in p['records']
                        for a,b in zip(r['logits'],baseline[p['delay_ms'],r['cue']]['logits']))
                record['interventions'][intervention]=value
        records.append(record)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=Path('/root/autodl-tmp/brain-sokoban-v1-20261003/studies/brain-sokoban-v1/runs'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=summarize(args.results_root)
    value=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output:args.output.write_text(value+'\n',encoding='utf-8')
    print(value)
