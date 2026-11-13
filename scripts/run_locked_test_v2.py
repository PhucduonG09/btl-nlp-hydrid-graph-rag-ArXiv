"""Prepare a held-out protocol without opening test QA; execute after human calibration.

No API calls. Exactly five fixed retrieval methods; no test-driven selection.
--check-dev compares the deployment engine against six existing DEV examples.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.human_calibration import ratings,resolved_ratings_v2
from src.eval.benchmark_metrics import summarize
BASE=ROOT/'results/benchmark_v2_assisted';OUT=BASE/'test_01'
DATA=ROOT/'data/eval/benchmark_v2_assisted'
METHODS=['bm25','dense','rrf','text_expansion','graph']
SOURCES=['scripts/run_locked_test_v2.py','src/retrieval/engine.py','src/retrieval/offline.py',
    'src/retrieval/graph.py','src/kg/mentions.py','src/kg/aliases.py',
    'src/eval/benchmark_metrics.py','src/eval/ir_metrics.py','src/generation/controlled.py',
    'src/eval/human_calibration.py']
def load(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def calibration_gate(root=ROOT):
    pack=root/'data/eval/generation_calibration_v2'
    manifest=json.loads((pack/'manifest.json').read_text(encoding='utf-8'))
    if sha(pack/'blind_items.jsonl')!=manifest['items_sha256']:raise ValueError('Blind calibration pack changed')
    items=load(pack/'blind_items.jsonl')
    resolved,pending=resolved_ratings_v2(ratings(pack/'ratings.sqlite3'),{r['sample_id'] for r in items})
    # Fail before opening qa_test; assistant/automatic labels never fill this gate.
    if len(items)!=50 or len(resolved)!=50:raise ValueError(f'Human calibration pending: {len(resolved)}/50. No test QA opened.')
    flagged=sum(1 for r in resolved.values() if r['reference_issue'])
    return dict(n_resolved=50,n_reference_issue_flagged=flagged,amendment='A1-2026-11-11',ratings_sha256=sha(pack/'ratings.sqlite3'),
        items_sha256=manifest['items_sha256'],policy='Named human ratings; no unrated, uncertain or disagreeing label. Amendment A1: reference_issue is recorded as a stratum flag and does not block when raters agree.')

def protocol():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    cfg=json.loads((BASE/'dev_01/selected_config.json').read_text(encoding='utf-8'))
    dense=json.loads((BASE/'dev_01/runtime.json').read_text(encoding='utf-8'))['dense']
    assert (cfg['bm25'],cfg['rrf'],cfg['graph'])==('bm25_k1.5_b0.75','rrf_k10','linked_k1_rrf_w2')
    return dict(version='fixed-test-retrieval-v1',methods=METHODS,
        primary_comparison=['graph','rrf'],primary_retrieval_endpoint='Bridge All@10',
        secondary_endpoints=['Single R@10','MRR@10','Hop1@10','Hop2@10'],
        selection_on_test='None. All five methods reported, no tuning or choosing a winner for a new test run.',
        parameters=dict(bm25_k1=1.5,bm25_b=.75,dense_revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a',
            rrf_k=10,linked_seeds=3,graph_hops=1,final_rrf_k=60,final_rrf_weights=[1,2]),
        test_qa_sha256=manifest['qa_sha256']['qa_test.jsonl'],corpus_sha256=manifest['corpus_sha256'],
        graph_sha256=cfg['graph_sha256'],dev_config_sha256=sha(BASE/'dev_01/selected_config.json'),
        document_embedding_cache=dict(path='data/index/emb_scale_6cdeb8a3dc.npy',sha256=dense['document_cache_sha256'],
            limitation=dense['historical_document_embedding_revision']),
        code_sha256={p:sha(ROOT/p) for p in SOURCES},
        preparation_test_access='None. Expected hash read from manifest; test QA file not opened.',
        execution_gate='50 actual human answer ratings resolved before first test QA load. Generation judge calibration must be assessed separately before paid held-out generation.',
        limitations=manifest['limitations'])

def check_dev():
    print('Loading fixed engine for six DEV queries (local models only)...',flush=True)
    from src.retrieval.engine import load_engine
    ids={j['id'] for j in load(BASE/'dev_01/generation_prepared/pilot_jobs.jsonl')}
    qa=[q for q in load(DATA/'qa_dev.jsonl') if q['id'] in ids]
    saved={r['id']:r for r in load(BASE/'dev_01/retrieval.jsonl')}
    names=dict(bm25='bm25_k1.5_b0.75',dense='dense',rrf='rrf_k10',graph='linked_k1_rrf_w2',text_expansion='linked_text_expansion')
    engine=load_engine();checks=[]
    for q in qa:
        got=engine.retrieve(q['question'],method='all')
        matched={name:got['rankings'][name]==saved[q['id']]['rankings'][source] for name,source in names.items()}
        checks.append(dict(id=q['id'],exact_top100_match=matched));print(q['id'],matched,flush=True)
    result=dict(n_questions=len(qa),n_methods=5,all_exact=all(all(r['exact_top100_match'].values()) for r in checks),checks=checks,
        test_access='None; six existing DEV pilot questions only.',api_calls=0,engine_sha256=sha(ROOT/'src/retrieval/engine.py'))
    dump(BASE/'dev_01/engine_consistency.json',result)
    assert len(qa)==6 and result['all_exact'],'Engine differs from recorded DEV pipeline; do not run test'

def execute():
    gate=calibration_gate()  # Before any test QA access.
    saved=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert saved==protocol(),'Protocol/code changed; reconcile before opening test, never silently retune'
    check=json.loads((BASE/'dev_01/engine_consistency.json').read_text(encoding='utf-8'))
    assert check['all_exact'] and check['engine_sha256']==sha(ROOT/'src/retrieval/engine.py')
    assert sha(ROOT/saved['document_embedding_cache']['path'])==saved['document_embedding_cache']['sha256'],'Embedding cache changed'
    assert not (OUT/'execution_complete.json').exists(),'Test completed already; do not rerun or tune.'
    # A receipt survives failures, preventing accidental fresh tuning after exposure.
    receipt=OUT/'test_opened.json'
    if receipt.exists():assert json.loads(receipt.read_text(encoding='utf-8'))['protocol_sha256']==sha(OUT/'protocol.json'),'Resume requires unchanged protocol'
    else:dump(receipt,dict(started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),protocol_sha256=sha(OUT/'protocol.json'),human_gate=gate))
    path=DATA/'qa_test.jsonl';assert sha(path)==saved['test_qa_sha256'];qa=load(path)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    assert sha(ROOT/manifest['corpus'])==saved['corpus_sha256']
    assert sha(ROOT/'data/kg/reference_mentions_v2_assisted.json')==saved['graph_sha256']
    from src.retrieval.engine import load_engine
    engine=load_engine();records=load(OUT/'retrieval.jsonl') if (OUT/'retrieval.jsonl').exists() else []
    done={r['id']:r for r in records};assert len(done)==len(records) and set(done)<={q['id'] for q in qa}
    rankings={m:{r['id']:r['rankings'][m] for r in records} for m in METHODS};start=time.perf_counter()
    for q in qa:
        if q['id'] in done:continue
        r=engine.retrieve(q['question'],method='all');record=dict(id=q['id'],**r)
        records.append(record)
        for method in METHODS:rankings[method][q['id']]=r['rankings'][method]
        with (OUT/'retrieval.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f"Fixed test {len(records)}/{len(qa)}",flush=True)
    dump(OUT/'metrics.json',{m:summarize(qa,r) for m,r in rankings.items()})
    dump(OUT/'runtime.json',dict(n_questions=len(qa),elapsed_seconds=time.perf_counter()-start,api_calls=0,configuration_selection=False))
    # Prepare generation sources, using the same fixed DEV prompt and context cap.
    from src.generation.controlled import prepare
    import tiktoken
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'data/cache/tiktoken');encoder=tiktoken.get_encoding('cl100k_base')
    jobs=[dict(id=q['id'],system=s,**prepare(q['question'],rankings[s][q['id']],engine.papers,encoder,budget=6000,top_k=10)) for q in qa for s in ['rrf','graph']]
    (OUT/'generation_jobs.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs),encoding='utf-8')
    dump(OUT/'generation_preparation.json',dict(n_jobs=len(jobs),api_calls=0,jobs_sha256=sha(OUT/'generation_jobs.jsonl'),status='Prepared only; generator/judge not executed.'))
    dump(OUT/'execution_complete.json',dict(protocol_sha256=sha(OUT/'protocol.json'),n_queries=len(qa),retrieval_sha256=sha(OUT/'retrieval.jsonl'),api_calls=0))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','check-dev','execute']);args=parser.parse_args()
    if args.action=='check-dev':check_dev();return
    if args.action=='execute':execute();return
    OUT.mkdir(exist_ok=True);p=OUT/'protocol.json';new=protocol()
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==new,'Existing protocol differs. Do not overwrite.'
    else:dump(p,new)
    print('Protocol prepared. No test QA read, no API calls.')
if __name__=='__main__':main()
