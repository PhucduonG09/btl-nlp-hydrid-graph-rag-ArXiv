"""Held-out TEST generation/judging, run ONCE under docs/protocol-test-v2.md (A1, A2).

Stages (explicit CLI; API calls only in generate/judge):
  generate   claude-sonnet-5 (same model/prompt/context cap as DEV), 56 jobs = 28 questions x {rrf, graph}
  judge      gpt-6-sol (same judge/prompt as DEV)
  summarize  offline; Amendment A2 metrics with 95% CI clustered by gold component
  summarize --dev-check   offline validation of the metric code on DEV records only (never opens qa_test)

Gates (all enforced before any model call): execute() finished (execution_complete.json), protocol hash unchanged,
50/50 human ratings resolved (A1) with the same ratings hash recorded at test opening, A2 present in the protocol doc,
models/prompt identical to the DEV protocol. No tuning: models, prompt, temperature and caps are read from DEV files.
Every attempt (including errors) is appended; --retry-errors only re-attempts failed jobs and is logged.
"""
import argparse,hashlib,json,os,re,sys,time
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_generation_v2 import JUDGE,normalized
BASE=ROOT/'results/benchmark_v2_assisted';OUT=BASE/'test_01';GEN=OUT/'generation'
DEVGEN=BASE/'dev_01/generation_prepared/claude5_controlled'
DATA=ROOT/'data/eval/benchmark_v2_assisted'
SEED=20261111
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def gates():
    from src.eval.human_calibration import ratings,resolved_ratings_v2
    from src.generation.controlled import VERSION
    done=OUT/'execution_complete.json'
    if not done.exists():raise SystemExit('Run run_locked_test_v2.py execute first (retrieval test must be complete).')
    proto=OUT/'protocol.json';receipt=json.loads((OUT/'test_opened.json').read_text(encoding='utf-8'))
    assert json.loads(done.read_text(encoding='utf-8'))['protocol_sha256']==sha(proto)==receipt['protocol_sha256'],'Protocol changed since execute'
    prep=json.loads((OUT/'generation_preparation.json').read_text(encoding='utf-8'))
    assert sha(OUT/'generation_jobs.jsonl')==prep['jobs_sha256'],'generation_jobs.jsonl changed'
    pack=ROOT/'data/eval/generation_calibration_v2'
    items=load(pack/'blind_items.jsonl')
    resolved,_=resolved_ratings_v2(ratings(pack/'ratings.sqlite3'),{r['sample_id'] for r in items})
    assert len(resolved)==50 and sha(pack/'ratings.sqlite3')==receipt['human_gate']['ratings_sha256'],'Human ratings changed or incomplete since test opening'
    doc=(ROOT/'docs/protocol-test-v2.md').read_text(encoding='utf-8')
    assert 'Amendment A2' in doc,'Amendment A2 (generation metrics) must be recorded before generation'
    g=json.loads((DEVGEN/'generation_protocol.json').read_text());j=json.loads((DEVGEN/'judge_protocol.json').read_text())
    assert g['prompt_version']==VERSION=='abstract-rag-v2'
    return g,j,proto

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['generate','judge','summarize'])
    ap.add_argument('--workers',type=int,default=2);ap.add_argument('--retry-errors',action='store_true');ap.add_argument('--dev-check',action='store_true')
    args=ap.parse_args()
    if args.stage=='summarize':return summarize_stage(args.dev_check)
    g,j,proto=gates();GEN.mkdir(exist_ok=True)
    assert not (GEN/'generation_complete.json').exists() or args.stage!='generate','Test generation already complete; do not rerun.'
    judging=args.stage=='judge';model=j['model'] if judging else g['model'];max_out=j['max_output_tokens'] if judging else g['max_output_tokens']
    if ('claude' in g['model'].lower())==('claude' in j['model'].lower()):raise SystemExit('Judge must be a different model family.')
    from src.common import llm
    from src.kg.llm_extract import parse_json
    from src.kg.evidence import locate_evidence
    from src.generation.controlled import citation_diagnostics
    from src.eval.run_status import read_records
    qa_path=DATA/'qa_test.jsonl';saved=json.loads(proto.read_text(encoding='utf-8'))
    assert sha(qa_path)==saved['test_qa_sha256'];qa={q['id']:q for q in load(qa_path)}
    jobs=load(OUT/'generation_jobs.jsonl');assert len(jobs)==2*len(qa) and all(x['id'] in qa for x in jobs)
    out=GEN/('judge.jsonl' if judging else 'generation.jsonl')
    pf=GEN/('judge_protocol.json' if judging else 'generation_protocol.json')
    p=dict(model=model,temperature=0,max_output_tokens=max_out,prompt_version=j['prompt_version'] if judging else g['prompt_version'],
           jobs_sha256=sha(OUT/'generation_jobs.jsonl'),protocol_sha256=sha(proto),source_dev_protocol_sha256=sha(DEVGEN/('judge_protocol.json' if judging else 'generation_protocol.json')),
           amendments=['A1','A2'],tuning='None. Models/prompt/caps copied from DEV protocol.')
    if pf.exists():assert json.loads(pf.read_text(encoding='utf-8'))==p,'Protocol changed; do not retune on test'
    else:dump(pf,p)
    records,bad=read_records(out);done={}
    for r in records:done[(r['id'],r['system'])]=r   # latest attempt wins; every attempt stays in the file
    generated={}
    if judging:
        grecs,_=read_records(GEN/'generation.jsonl');generated={(r['id'],r['system']):r for r in grecs}
        assert all((x['id'],x['system']) in generated and not generated[(x['id'],x['system'])].get('error') for x in jobs),'Complete generation first'
    llm.CACHE_PATH=GEN/('judge_llm_cache.jsonl' if judging else 'generator_llm_cache.jsonl');llm._cache=None;llm._load_cache()
    def work(job):
        key=(job['id'],job['system']);q=qa[job['id']]
        rec=dict(id=key[0],system=key[1],type=q['type'],requested_model=model,attempt_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 source_job_sha256=hashlib.sha256(json.dumps(job,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
        if judging:
            gen=generated[key]
            prompt=f"Question: {q['question']}\nReference answer (may be imperfect): {q['reference_answer']}\nCandidate answer: {gen['answer']}\nSources:\n"+'\n\n'.join(f"[{s['number']}] arXiv:{s['paper_id']}\n{s['title']}\n{s['abstract']}" for s in job['sources'])
            messages=[dict(role='system',content=JUDGE),dict(role='user',content=prompt)]
        else:messages=job['messages']
        retrying=args.retry_errors and key in done and done[key].get('error')
        text,usage=llm.chat(messages,model=model,temperature=0,max_tokens=max_out,run_id='benchmark-v2-test-judge-v1' if judging else 'benchmark-v2-test-generation-v1',use_cache=not retrying)
        rec.update(usage=usage,served_model=usage.get('served_model'),model_label_mismatch=normalized(usage.get('served_model'))!=normalized(model))
        if not judging:
            ctx={s['paper_id'] for s in job['sources']}
            rec.update(answer=text,context_paper_ids=sorted(ctx),gold_all_in_context=set(q['gold_ids'])<=ctx,**citation_diagnostics(text,job['sources']));return rec
        parsed=parse_json(text);rec['raw_judge']=text
        if not isinstance(parsed,dict) or parsed.get('quality') not in {'correct','partial','incorrect','abstained'} or not isinstance(parsed.get('claims'),list):
            rec['error']='invalid_judge_json_or_schema';return rec
        rel=parsed.get('relevance')
        if type(rel) is not int or not 1<=rel<=5:rec['error']='invalid_judge_relevance';return rec
        smap={s['number']:s for s in job['sources']};val=[]
        for c in parsed['claims']:
            if not isinstance(c,dict) or not isinstance(c.get('text'),str) or type(c.get('supported')) is not bool:rec['error']='invalid_judge_claim';return rec
            n=c.get('source_number');qt=c.get('evidence_quote','')
            if (n is not None and type(n) is not int) or not isinstance(qt,str):rec['error']='invalid_judge_source_or_quote';return rec
            s=smap.get(n);lit=bool(s and qt.strip() and locate_evidence(s['title']+'\n'+s['abstract'],qt));val.append(dict(c,evidence_quote_valid=lit))
        factual=bool(val) and not gen['refused']
        rec.update(quality=parsed['quality'],relevance=rel,claims=val,faithfulness_raw=sum(c['supported'] for c in val)/len(val) if factual else None,
                   supported_claims_with_invalid_quote=sum(c['supported'] and not c['evidence_quote_valid'] for c in val),human_calibrated=False);return rec
    todo=[x for x in jobs if (x['id'],x['system']) not in done or (args.retry_errors and done[(x['id'],x['system'])].get('error'))]
    failures=0;t0=time.perf_counter();usages=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut={ex.submit(work,x):x for x in todo}
        for f in as_completed(fut):
            x=fut[f]
            try:rec=f.result()
            except Exception as e:rec=dict(id=x['id'],system=x['system'],type=qa[x['id']]['type'],error=type(e).__name__,detail=str(e)[:200])
            failures+=bool(rec.get('error'))
            with out.open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,ensure_ascii=False)+'\n')
            done[(x['id'],x['system'])]=rec
            if 'usage' in rec:usages.append(rec['usage'])
            n=sum((y['id'],y['system']) in done for y in jobs)
            if n%4==0 or n==len(jobs):print(f'{args.stage}: {n}/{len(jobs)}, errors={failures}',flush=True)
    sel=[done[(x['id'],x['system'])] for x in jobs];errs=sum(bool(r.get('error')) for r in sel)
    dump(GEN/f'{args.stage}_runtime.json',dict(stage=args.stage,n_jobs=len(jobs),n_new_attempts=len(todo),errors_now=errs,retry_errors=args.retry_errors,
        usage_this_invocation=llm.cost_summary(usages),served_labels=dict(Counter(r.get('served_model','missing') for r in sel)),
        label_mismatches=sum(bool(r.get('model_label_mismatch')) for r in sel),elapsed_seconds=time.perf_counter()-t0))
    if not judging and errs==0:dump(GEN/'generation_complete.json',dict(jobs_sha256=p['jobs_sha256']))
    print('errors remaining:',errs,'(re-run with --retry-errors for transient failures)' if errs else '')

# ---------------------------------------------------------------- summarize (offline)
def components(qa):
    parent={}
    def find(x):
        while parent.setdefault(x,x)!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for q in qa.values():
        ids=q['gold_ids']
        for a in ids[1:]:parent[find(a)]=find(ids[0])
        find(ids[0])
    return {q['id']:find(q['gold_ids'][0]) for q in qa.values()}
def rate(rows,pred,denom=lambda r:True):
    d=[r for r in rows if denom(r)];return (sum(pred(r) for r in d),len(d))
def summarize_stage(dev_check):
    import numpy as np
    if dev_check:
        qa={q['id']:q for q in load(DATA/'qa_dev.jsonl')};gen=load(DEVGEN/'generation.jsonl');jud=load(DEVGEN/'judge.jsonl');outp=DEVGEN/'a2_metrics_devcheck.json'
    else:
        assert (GEN/'judge_runtime.json').exists(),'Run judge first'
        qa={q['id']:q for q in load(DATA/'qa_test.jsonl')};gen=load(GEN/'generation.jsonl');jud=load(GEN/'judge.jsonl');outp=GEN/'a2_metrics.json'
    G={(r['id'],r['system']):r for r in gen if not r.get('error')};J={(r['id'],r['system']):r for r in jud if not r.get('error')}
    comp=components(qa);rows=[]
    for (i,s),g in G.items():
        if (i,s) in J and i in qa:rows.append(dict(id=i,system=s,type=qa[i]['type'],comp=comp[i],refused=g['refused'],ctx=g['gold_all_in_context'],q=J[(i,s)]['quality'],
            faith=J[(i,s)].get('faithfulness_raw')))
    ans=lambda r:r['q'] in ('correct','partial')
    def metrics(sub):
        out={}
        for s in ('rrf','graph'):
            r=[x for x in sub if x['system']==s];f=[x['faith'] for x in r if x['faith'] is not None]
            out[s]=dict(n=len(r),refused=sum(x['refused'] for x in r),answered_substantive=sum(ans(x) for x in r),
                gold_in_context=sum(x['ctx'] for x in r),answered_given_gold_in_context=rate(r,ans,lambda x:x['ctx']),
                faithfulness_macro_nonabstaining=float(np.mean(f)) if f else None,faithfulness_answers=len(f),
                correct_rate_judge_only_sensitivity=sum(x['q']=='correct' for x in r)/len(r) if r else None)
        return out
    def cluster_ci(sub,fn,B=5000):
        rng=np.random.default_rng(SEED);by=defaultdict(list)
        for x in sub:by[x['comp']].append(x)
        keys=list(by)
        if not keys:return None
        vals=[]
        for _ in range(B):
            pick=[y for k in rng.choice(len(keys),len(keys)) for y in by[keys[k]]];v=fn(pick)
            if v is not None:vals.append(v)
        return [float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))] if vals else None
    def diff(sub,key):
        def f(rs):
            a=[x for x in rs if x['system']=='graph'];b=[x for x in rs if x['system']=='rrf']
            if not a or not b:return None
            return key(a)-key(b)
        return f
    res=dict(scope='Amendment A2 metrics. Primary: refusal, substantive-answer rate (correct+partial), conditional on gold in context, faithfulness. correct_rate = judge sensitivity only.',
             judge_caveat='Judge calibration (DEV, n=50): quality agreement 76%, kappa 0.58; judge stricter than humans; partial/incorrect not validated.',
             n_questions=len(qa),n_rows=len(rows),dev_check=dev_check,bootstrap=dict(B=5000,seed=SEED,cluster='gold-paper connected component'),groups={})
    for name,sub in [('all',rows),('single',[r for r in rows if r['type']=='single']),('bridge',[r for r in rows if r['type']=='bridge'])]:
        m=metrics(sub)
        sub_ans=diff(sub,lambda a:np.mean([ans(x) for x in a]));sub_ctx=diff(sub,lambda a:np.mean([x['ctx'] for x in a]))
        m['graph_minus_rrf']=dict(substantive_answer_rate=dict(point=sub_ans(sub),ci95=cluster_ci(sub,sub_ans)),
            gold_in_context_rate=dict(point=sub_ctx(sub),ci95=cluster_ci(sub,sub_ctx)),
            note='Difference in answering is mediated by retrieval (gold in context); do not attribute to the generation step.')
        res['groups'][name]=m
    dump(outp,res);print(json.dumps({k:{s:{kk:vv for kk,vv in v[s].items() if kk in('n','refused','answered_substantive','gold_in_context')} for s in('rrf','graph')} for k,v in res['groups'].items()},indent=1,ensure_ascii=False))
    print('written',outp)
if __name__=='__main__':main()
