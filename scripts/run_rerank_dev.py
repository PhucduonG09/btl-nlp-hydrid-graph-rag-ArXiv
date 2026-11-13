"""Pinned cross-encoder on DEV; matched RRF/graph controls and resumable scores."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.retrieval.rerank import reorder_candidates
from src.retrieval.graph import fuse_rrf
from src.eval.benchmark_metrics import summarize,selection_score
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    dataset=ROOT/'data/eval/benchmark_v2_assisted';base=ROOT/'results/benchmark_v2_assisted/dev_01'
    out=base/'rerank';out.mkdir(exist_ok=True)
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    assert sha(dataset/'qa_dev.jsonl')==manifest['qa_sha256']['qa_dev.jsonl']
    assert sha(ROOT/manifest['corpus'])==manifest['corpus_sha256']
    qa=load(dataset/'qa_dev.jsonl');papers={p['paper_id']:p for p in load(ROOT/manifest['corpus'])}
    source={r['id']:r for r in load(base/'retrieval.jsonl')};selected=json.loads((base/'selected_config.json').read_text())
    assert set(source)=={q['id'] for q in qa}
    model_manifest_path=ROOT/'data/models/reranker_manifest.json'
    model_info=json.loads(model_manifest_path.read_text(encoding='utf-8'))
    model_path=Path(model_info['local_path']);assert model_path.name==model_info['revision']
    protocol=dict(model=model_info,depth=50,max_length=512,batch_size=16,device='cpu',rrf_mix_weights=[.25,.5,1,2],rrf_k=60,
                  qa_dev_sha256=sha(dataset/'qa_dev.jsonl'),source_retrieval_sha256=sha(base/'retrieval.jsonl'),
                  source_config_sha256=sha(base/'selected_config.json'),
                  code_sha256={p:sha(ROOT/p) for p in ['scripts/run_rerank_dev.py','src/retrieval/rerank.py','src/retrieval/graph.py']},
                  model_weight_sha256=sha(model_path/'model.safetensors'),test_access='None. DEV only.',
                  selection='Balanced single/bridge All@10, then balanced MRR@10. Include no-rerank baseline as an option.',
                  scoring='Original question paired with title+abstract. Score each query/document once across the union of two top-50 pools. No gold/bridge entity in scoring input.')
    protocol_path=out/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text(encoding='utf-8'))==protocol,'Protocol changed; use a new rerank run directory'
    else:dump(protocol_path,protocol)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    model=CrossEncoder(str(model_path),device='cpu',max_length=512,local_files_only=True,trust_remote_code=False)
    cache_path=out/'scores.jsonl';cached={r['key']:r for r in load(cache_path)} if cache_path.exists() else {}
    rankings={};records=[];latency=[];new_pairs=0;start=time.perf_counter()
    for i,q in enumerate(qa):
        qid=q['id'];originals=dict(rrf=source[qid]['rankings'][selected['rrf']],graph=source[qid]['rankings'][selected['graph']])
        pool=list(dict.fromkeys(originals['rrf'][:50]+originals['graph'][:50]))
        texts=[papers[pid]['title']+'. '+papers[pid]['abstract'] for pid in pool]
        key=hashlib.sha256(json.dumps([model_info['revision'],512,q['question'],list(zip(pool,texts))],ensure_ascii=False).encode()).hexdigest()
        if key in cached:record=cached[key]
        else:
            t=time.perf_counter();values=np.asarray(model.predict([(q['question'],text) for text in texts],batch_size=16,show_progress_bar=False)).reshape(-1)
            assert len(values)==len(pool) and np.isfinite(values).all()
            record=dict(id=qid,key=key,scores=dict(zip(pool,values.astype(float).tolist())),scoring_seconds=time.perf_counter()-t,n_pairs=len(pool))
            with cache_path.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            new_pairs+=len(pool)
        latency.append(record['scoring_seconds']);variants={}
        for family,ranked in originals.items():
            variants[family+'_base']=ranked
            ce=reorder_candidates(ranked,record['scores'],depth=50);variants[family+'_ce']=ce
            for w in [.25,.5,1,2]:variants[family+f'_ce_mix_w{w}']=fuse_rrf([ranked,ce],weights=[1,w])
        for name,ranking in variants.items():rankings.setdefault(name,{})[qid]=ranking
        records.append(dict(id=qid,rankings=variants))
        if (i+1)%10==0 or i+1==len(qa):print(f'Rerank DEV {i+1}/{len(qa)}',flush=True)
    metrics={name:summarize(qa,ranking) for name,ranking in rankings.items()}
    best={family:sorted([m for m in metrics if m.startswith(family+'_')],key=lambda m:(-selection_score(metrics[m])[0],-selection_score(metrics[m])[1],m))[0] for family in ['rrf','graph']}
    dump(out/'metrics.json',metrics);dump(out/'selected_config.json',dict(selected=best,protocol_sha256=sha(protocol_path),test_evaluated=False))
    (out/'retrieval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    dump(out/'runtime.json',dict(elapsed_seconds=time.perf_counter()-start,new_pairs=new_pairs,n_queries=len(qa),
        p50_union_scoring_seconds=float(np.percentile(latency,50)),p95_union_scoring_seconds=float(np.percentile(latency,95)),
        latency_scope='Per-query union of RRF and graph top-50 pools, including tokenizer/inference, excluding initial retrieval/model load. Cached scores retain original measured latency.',network='Offline after model download; no paid API.'))
    print(json.dumps(dict(selected=best,bridge_all10={m:round(v['bridge']['All@10'],4) for m,v in metrics.items()},single_R10={m:round(v['single']['R@10'],4) for m,v in metrics.items()}),indent=2),flush=True)
if __name__=='__main__':main()
