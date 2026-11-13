"""Build reviewable DEV generation inputs and a pilot plan. No API calls."""
import hashlib,json,os,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.generation.controlled import prepare,VERSION
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def main():
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'data/cache/tiktoken')
    import tiktoken
    encoder=tiktoken.get_encoding('cl100k_base')
    dataset=ROOT/'data/eval/benchmark_v2_assisted';base=ROOT/'results/benchmark_v2_assisted/dev_01'
    m=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    assert sha(dataset/'qa_dev.jsonl')==m['qa_sha256']['qa_dev.jsonl']
    assert sha(ROOT/m['corpus'])==m['corpus_sha256']
    qa=load(dataset/'qa_dev.jsonl');papers={p['paper_id']:p for p in load(ROOT/m['corpus'])}
    runs={r['id']:r for r in load(base/'retrieval.jsonl')};cfg=json.loads((base/'selected_config.json').read_text())
    assert set(runs)=={q['id'] for q in qa}
    systems={'rrf':cfg['rrf'],'graph':cfg['graph']}
    out=base/'generation_prepared';out.mkdir(exist_ok=True)
    jobs=[]
    for q in qa:
        for label,method in systems.items():
            payload=prepare(q['question'],runs[q['id']]['rankings'][method],papers,encoder,budget=6000,top_k=10)
            jobs.append(dict(id=q['id'],system=label,retrieval_method=method,**payload))
    rng=random.Random(42);pilot_ids=set()
    for kind in ['single','bridge']:pilot_ids.update(q['id'] for q in rng.sample([q for q in qa if q['type']==kind],3))
    pilot=[j for j in jobs if j['id'] in pilot_ids]
    for filename,items in [('jobs_dev.jsonl',jobs),('pilot_jobs.jsonl',pilot)]:
        (out/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    report=dict(status='PREPARED_NOT_EXECUTED',n_dev_questions=len(qa),systems=systems,n_jobs=len(jobs),pilot_question_ids=sorted(pilot_ids),pilot_jobs=len(pilot),
                prompt_version=VERSION,context_budget_proxy_tokens=6000,top_k=10,max_output_tokens=768,temperature=0,
                token_counter='cl100k_base proxy. Counts message content, not provider-specific tokenization/chat overhead. Actual API usage must be logged at execution.',
                dev_input_proxy_tokens=sum(j['input_content_proxy_tokens'] for j in jobs),pilot_input_proxy_tokens=sum(j['input_content_proxy_tokens'] for j in pilot),
                dev_max_output_tokens=len(jobs)*768,pilot_max_output_tokens=len(pilot)*768,
                qa_dev_sha256=sha(dataset/'qa_dev.jsonl'),source_retrieval_sha256=sha(base/'retrieval.jsonl'),
                jobs_sha256=sha(out/'jobs_dev.jsonl'),pilot_jobs_sha256=sha(out/'pilot_jobs.jsonl'),
                paid_calls=0,estimated_cost=None,pricing='Hill will supply provider input/output rates later. No prices invented; no generation or judge calls made.',
                provenance='Assisted-source-reviewed experimental QA. Reference answers and gold IDs are absent from generator messages. Prepared DEV only; test not loaded.',
                execution_gates=['Confirm provider rates and model requested/served labels on pilot before scaling.','Keep same generator/prompt/context cap across systems.','Do not infer faithfulness from valid citation syntax.','Human judge calibration still required.'])
    (out/'plan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','n_jobs','pilot_jobs','dev_input_proxy_tokens','pilot_input_proxy_tokens','paid_calls']},indent=2))
if __name__=='__main__':main()
