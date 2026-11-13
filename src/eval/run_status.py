"""Read-only progress and provider-reported usage, including retry attempts."""
import json
from collections import Counter

def read_records(path):
    if not path.exists():return [],0
    records=[];bad=0
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('Not an object')
            records.append(row)
        except (ValueError,TypeError):bad+=1
    return records,bad

def summarize_records(records):
    latest={(r.get('id'),r.get('system')):r for r in records}
    usages=[r['usage'] for r in records if r.get('usage')]
    billed=[u for u in usages if not u.get('cached')]
    return dict(records=len(records),unique_jobs=len(latest),completed=sum(not r.get('error') for r in latest.values()),
                errors=sum(bool(r.get('error')) for r in latest.values()),
                provider_input_tokens=sum(u.get('in',0) for u in billed),provider_output_tokens=sum(u.get('out',0) for u in billed),
                cache_hits=sum(bool(u.get('cached')) for u in usages),
                served_models=dict(Counter(r.get('served_model',r.get('usage',{}).get('served_model','missing')) for r in records)),
                model_label_mismatches=sum(bool(r.get('model_label_mismatch')) for r in latest.values()))
