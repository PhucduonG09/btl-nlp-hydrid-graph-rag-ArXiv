"""Run capped, resumable DEV-only extraction using the repo's existing API config.

The five-paper pilot must be reviewed before processing the remaining papers.
No QA answers or gold bridge edges are sent to the extractor.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dotenv import dotenv_values
from src.common import llm
from src.kg.llm_extract import load_done,run_extraction,_ws

def load(path):
    with path.open(encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','full'],default='pilot')
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--retry-same-config',action='store_true');args=ap.parse_args()
    cfg=dotenv_values(ROOT/'.env')
    # Keep the API key and destination from the same explicitly configured file.
    for key in ['LLM_API_KEY','LLM_BASE_URL','LLM_MODEL']:
        if cfg.get(key):os.environ[key]=cfg[key]
    model=os.environ.get('LLM_MODEL');assert model and os.environ.get('LLM_API_KEY'),'Repo API configuration is incomplete'
    from openai import OpenAI,AuthenticationError,PermissionDeniedError
    client=OpenAI(base_url=os.environ.get('LLM_BASE_URL'),api_key=os.environ['LLM_API_KEY'].strip(),timeout=60,max_retries=0,
                  default_headers={'Accept-Encoding':'identity'})
    def create(**kwargs):
        try:return client.chat.completions.create(**kwargs)
        except AuthenticationError:raise llm.FatalLLMError('HTTP 401: authentication rejected') from None
        except PermissionDeniedError:raise llm.FatalLLMError('HTTP 403: permission denied') from None
    llm._client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    inputs=ROOT/'data/processed'/('bridge_v1_dev_kg_pilot5.jsonl' if args.stage=='pilot' else 'bridge_v1_dev_missing_kg_papers.jsonl')
    output=ROOT/'data/kg/bridge_dev_v1_extract_v3.jsonl'
    pilot_snapshot=ROOT/'data/kg/bridge_dev_v1_pilot5_v3.jsonl'
    outdir=ROOT/'results/bridge_v1';outdir.mkdir(parents=True,exist_ok=True)
    config_hash=sha(ROOT/'.env')
    previous_report=outdir/'extraction_pilot_runtime.json'
    if args.stage=='pilot' and previous_report.exists() and not args.retry_same_config:
        previous=json.loads(previous_report.read_text(encoding='utf-8'))
        if previous.get('config_file_sha256')==config_hash and previous.get('errors') and all(e.get('http_status')=='401' for e in previous['errors']):
            print('Configuration file is unchanged since the last HTTP 401. Save updated credentials before retrying; no API call was made.',flush=True)
            return 1
    if args.stage=='full':
        review=json.loads((outdir/'extraction_pilot_review.json').read_text(encoding='utf-8'))
        assert review['approved_for_full'] and review['pilot_sha256']==sha(pilot_snapshot),'Pilot has not passed review'
    papers=load(inputs);store={p['paper_id']:p for p in papers};before=load_done(output)
    pending={p['paper_id'] for p in papers if p['paper_id'] not in before or 'error' in before[p['paper_id']]}
    print(f"Stage {args.stage}: {len(papers)} papers; {len(pending)} pending; model={model}; max_output_tokens=1800",flush=True)
    start=time.perf_counter()
    if args.stage=='pilot':
        records,errors=run_extraction(papers[:1],output,prompt_version='v3',model=model,workers=1,max_tokens=1800)
        attempted=1
        if records and not errors and 'error' not in records[0]:
            more,more_errors=run_extraction(papers[1:],output,prompt_version='v3',model=model,workers=args.workers,max_tokens=1800)
            records+=more;errors += [(i+1,msg) for i,msg in more_errors];attempted=len(papers)
    else:
        records,errors=run_extraction(papers,output,prompt_version='v3',model=model,workers=args.workers,max_tokens=1800)
        attempted=len(papers)
    # Never print raw SDK/authentication errors; expose only paper ID and status.
    safe_errors=[]
    for index,message in errors:
        status=re.search(r'HTTP (\d{3})',message)
        item=dict(paper_id=papers[index]['paper_id'],error_type=message.split('(',1)[0],http_status=status.group(1) if status else None)
        if item['error_type']=='TypeError':
            detail=message.replace(os.environ['LLM_API_KEY'],'[REDACTED]')
            detail=re.sub(r'\bsk-[A-Za-z0-9_-]+','[REDACTED]',detail)
            item['detail']=detail[:250]
        safe_errors.append(item)
    violations=[]
    for r in records:
        text=f"Title: {store[r['paper_id']]['title']}\nAbstract: {store[r['paper_id']]['abstract']}"
        for edge in r.get('relations',[]):
            if not edge.get('evidence') or _ws(edge['evidence']) not in _ws(text):violations.append(r['paper_id'])
    new_records=[r for r in records if r['paper_id'] in pending]
    report=dict(stage=args.stage,requested_model=model,n_input=len(papers),n_attempted=attempted,n_complete=len(records),config_file_sha256=config_hash,
                new_complete=len(new_records),n_relations=sum(len(r.get('relations',[])) for r in records),
                invalid_json=sum('error' in r for r in records),evidence_violations=violations,
                rejected_no_evidence=sum(r.get('dropped_no_evidence',0) for r in records),
                invalid_items=sum(r.get('n_invalid_items',0) for r in records),
                missing_ids=sorted(set(store)-{r['paper_id'] for r in records}),errors=safe_errors,
                usage_this_stage=llm.cost_summary([r['usage'] for r in new_records]),
                elapsed_seconds=round(time.perf_counter()-start,2),
                scope='DEV gold-conditioned coverage diagnostic; not the final full-corpus KG.')
    (outdir/f'extraction_{args.stage}_runtime.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.stage=='pilot' and not report['missing_ids'] and not report['invalid_json'] and not violations:
        pilot_snapshot.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if errors or violations or report['missing_ids'] or report['invalid_json']:return 1
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as error:
        # Intentionally do not echo exception bodies, which can include provider
        # messages or authentication details.
        print(json.dumps(dict(error_type=type(error).__name__,message='Extraction did not complete; no credentials were printed.')))
        sys.exit(1)
