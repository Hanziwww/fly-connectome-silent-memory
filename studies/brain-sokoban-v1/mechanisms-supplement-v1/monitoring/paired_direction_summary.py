"""Descriptive first-push direction audit; no model loading or parameter fitting."""
from pathlib import Path
import csv
import json

root=Path(__file__).resolve().parents[1]
models=json.loads((root/'models.json').read_text(encoding='utf-8'))
rows=[]
for model in models:
    name=model['id'];job='selective-'+name
    receipt=root/'receipts'/(job+'.json')
    if not receipt.exists() or not json.loads(receipt.read_text())['transport_verified']:continue
    out=root/'runs'/job/'conditions'
    for delay in (40,160,640):
        for prefix in (0,4):
            stem=f'd{delay}-p{prefix}'
            baseline=json.loads((out/(stem+'-baseline.json')).read_text())['rows']
            for kind in ('cue','scrambled'):
                treated=json.loads((out/(stem+f'-read-continuous-1.0-{kind}.json')).read_text())['rows']
                for actual,recipient in zip(treated,baseline):
                    donor=next(r for r in baseline if r['cue']==1-recipient['cue'] and
                               r['context']==recipient['context'] and r['visible']==recipient['visible'])
                    eligible=recipient['status']=='solved' and donor['status']=='solved'
                    rows.append(dict(model=name,delay=delay,prefix=prefix,kind=kind,case=actual['case'],
                        visible=actual['visible'],context=actual['context'],eligible_both_baselines_solved=eligible,
                        own_push=recipient.get('first_push_action'),donor_push=donor.get('first_push_action'),
                        actual_push=actual.get('first_push_action'),solved=actual['status']=='solved',
                        follows_donor=actual.get('first_push_action')==donor.get('first_push_action'),
                        follows_recipient=actual.get('first_push_action')==recipient.get('first_push_action')))
if rows:
    folder=root/'analysis';folder.mkdir(exist_ok=True)
    with (folder/'directional-swap-audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={}
    for r in rows:
        if not r['eligible_both_baselines_solved']:continue
        k=(r['model'],r['visible'],r['kind'])
        g=groups.setdefault(k,dict(model=k[0],visible=k[1],kind=k[2],eligible=0,follows_donor=0,follows_recipient=0,solved=0))
        g['eligible']+=1
        for field in ('follows_donor','follows_recipient','solved'):g[field]+=int(r[field])
    report=dict(analysis='Descriptive audit restricted to pairs whose two baseline cue histories solve',
        restriction_uses_baseline_only=True,independent_map_replicates=False,
        primary_all_history_outcomes='condition-outcomes.csv',groups=list(groups.values()))
    (folder/'directional-swap-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
