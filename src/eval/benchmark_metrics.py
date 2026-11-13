"""Mixed single/bridge metrics; balanced selection prevents single-count dominance."""
import numpy as np
from src.eval.ir_metrics import METRICS

def summarize(rows,rankings):
    result={}
    for group,selected in [('all',rows),('single',[r for r in rows if r['type']=='single']),('bridge',[r for r in rows if r['type']=='bridge'])]:
        if not selected:continue
        value=dict(n=len(selected))
        value.update({key:float(np.mean([metric(rankings[r['id']],r['gold_ids']) for r in selected])) for key,metric in METRICS.items()})
        if group=='bridge':
            for hop in [1,2]:value[f'Hop{hop}@10']=float(np.mean([r[f'hop{hop}_id'] in rankings[r['id']][:10] for r in selected]))
        result[group]=value
    return result

def selection_score(summary):
    groups=[summary[k] for k in ['single','bridge'] if k in summary]
    return (float(np.mean([g['All@10'] for g in groups])),float(np.mean([g['MRR@10'] for g in groups])))
