"""Offline DEV-only tuning and ablation. Never loads qa_test or qa_all."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.retrieval.offline import BM25Index
from src.retrieval.graph import GraphSearch,fuse_rrf,fuse_weighted
from src.kg.mentions import MentionIndex
from src.eval.benchmark_metrics import summarize,selection_score
from scripts.run_baselines import PREFIX

def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='benchmark_v2_assisted');ap.add_argument('--run-id',default='dev_01')
    args=ap.parse_args();base=ROOT/'data/eval'/args.dataset
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    qa_path=base/'qa_dev.jsonl';assert sha(qa_path)==manifest['qa_sha256']['qa_dev.jsonl']
    qa=load(qa_path);corpus_path=ROOT/manifest['corpus'];assert sha(corpus_path)==manifest['corpus_sha256']
    papers=load(corpus_path);ids=[p['paper_id'] for p in papers];assert len(ids)==len(set(ids))
    assert all(set(q['gold_ids'])<=set(ids) for q in qa)
    graph_path=ROOT/'data/kg/reference_mentions_v2_assisted.json';graph=json.loads(graph_path.read_text(encoding='utf-8'))
    assert graph['corpus_sha256']==manifest['corpus_sha256']
    assert graph['build_source_sha256']==sha(ROOT/'src/kg/mentions.py')
    out=ROOT/'results'/args.dataset/args.run_id
    if out.exists():raise SystemExit('Run directory exists; choose another --run-id to preserve results.')
    out.mkdir(parents=True)
    code_paths=['scripts/run_benchmark_dev.py','src/retrieval/offline.py','src/retrieval/graph.py','src/kg/mentions.py','src/kg/aliases.py','src/eval/benchmark_metrics.py']
    protocol=dict(dataset=args.dataset,qa_dev_sha256=sha(qa_path),corpus_sha256=sha(corpus_path),graph_sha256=sha(graph_path),
                  code_sha256={p:sha(ROOT/p) for p in code_paths},test_access='None: only qa_dev loaded.',
                  bm25_grid=dict(k1=[.8,1.5,2.0],b=[.4,.75]),rrf_grid=[10,30,60,100],
                  graph=dict(seed_count=3,hops=[1,2,3],ppr_alpha=.85,rrf_graph_weights=[.5,1,2],weighted_rank_weights=[.5,.5]),
                  selection='Equal-weight mean of single and bridge All@10, then equal-weight MRR@10. Deterministic name order for remaining ties.',
                  caution=manifest['limitations'])
    dump(out/'protocol.json',protocol)
    start=time.perf_counter();index=BM25Index(papers);rankings={};summaries={}
    def remember(name,lists):
        rankings[name]={q['id']:r for q,r in zip(qa,lists)};summaries[name]=summarize(qa,rankings[name])
    bm_names=[]
    for k1 in [.8,1.5,2.0]:
        for b in [.4,.75]:
            name=f'bm25_k{k1}_b{b}';remember(name,[index.rank(q['question'],k1,b) for q in qa]);bm_names.append(name)
    best_bm=sorted(bm_names,key=lambda n:(-selection_score(summaries[n])[0],-selection_score(summaries[n])[1],n))[0]
    print('BM25 DEV selected '+best_bm,flush=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('HF_HOME','D:/Dev/cache/huggingface')
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(min(4,os.cpu_count() or 1));model='BAAI/bge-base-en-v1.5'
    source=load(ROOT/'data/processed/pilot_papers.jsonl')+load(ROOT/'data/processed/distractors_20000.jsonl')
    source_ids=[p['paper_id'] for p in source];key=hashlib.sha1(('|'.join(source_ids)+model).encode()).hexdigest()[:10]
    doc_path=ROOT/'data/index'/f'emb_scale_{key}.npy';D_source=np.load(doc_path,mmap_mode='r')
    assert D_source.shape==(len(source_ids),768) and len(set(source_ids))==len(source_ids)
    lookup={pid:i for i,pid in enumerate(source_ids)};D=np.asarray(D_source[[lookup[pid] for pid in ids]],dtype=np.float32)
    assert np.isfinite(D).all() and np.allclose(np.linalg.norm(D,axis=1),1,atol=1e-3)
    encoder=SentenceTransformer(model,device='cpu',local_files_only=True)
    enc_start=time.perf_counter()
    Q=encoder.encode([PREFIX+q['question'] for q in qa],batch_size=8,normalize_embeddings=True,show_progress_bar=True).astype('float32')
    encoding_seconds=time.perf_counter()-enc_start;np.save(out/'query_embeddings.npy',Q)
    assert Q.shape==(len(qa),768) and np.isfinite(Q).all()
    S=Q@D.T;remember('dense',[[ids[j] for j in np.argsort(-row,kind='stable')[:100]] for row in S])
    rrf_names=[]
    for k in [10,30,60,100]:
        name=f'rrf_k{k}';remember(name,[fuse_rrf([rankings[best_bm][q['id']],rankings['dense'][q['id']]],k=k) for q in qa]);rrf_names.append(name)
    best_rrf=sorted(rrf_names,key=lambda n:(-selection_score(summaries[n])[0],-selection_score(summaries[n])[1],n))[0]
    print('RRF DEV selected '+best_rrf,flush=True)
    mi=MentionIndex(graph);search=GraphSearch(mi);records=[];timings={};graph_methods=set()
    for i,q in enumerate(qa):
        qid=q['id'];base_rank=rankings[best_rrf][qid];record=dict(id=qid,seeds={},paths={})
        for strategy in ['rrf','linked']:
            seeds=base_rank[:3] if strategy=='rrf' else mi.linked_seeds(q['question'],base_rank,count=3)
            record['seeds'][strategy]=seeds
            # Text expansion is a separate two-stage control with the same seeds.
            expansion=[index.rank(q['question']+' '+papers[ids.index(pid)]['title']+'. '+papers[ids.index(pid)]['abstract']) for pid in seeds]
            name=strategy+'_text_expansion';rankings.setdefault(name,{})[qid]=fuse_rrf([base_rank]+expansion)
            for kind in ['k1','k2','k3','ppr']:
                t=time.perf_counter()
                gr,paths=search.ppr(seeds) if kind=='ppr' else search.expand(seeds,hops=int(kind[1:]))
                key=strategy+'_'+kind;timings.setdefault(key,[]).append(time.perf_counter()-t)
                record['paths'][key]={pid:paths[pid] for pid in gr[:20] if pid in paths}
                for weight in [.5,1,2]:
                    name=key+f'_rrf_w{weight}';graph_methods.add(name)
                    rankings.setdefault(name,{})[qid]=fuse_rrf([base_rank,gr],weights=[1,weight])
                if kind!='ppr':
                    name=key+'_weighted';graph_methods.add(name)
                    rankings.setdefault(name,{})[qid]=fuse_weighted([base_rank,gr],[.5,.5])
        records.append(record)
        if (i+1)%10==0 or i+1==len(qa):print(f'Graph DEV {i+1}/{len(qa)}',flush=True)
    for method,ranking in rankings.items():summaries[method]=summarize(qa,ranking)
    best_graph=sorted(graph_methods,key=lambda n:(-selection_score(summaries[n])[0],-selection_score(summaries[n])[1],n))[0]
    for record in records:record['rankings']={m:r[record['id']] for m,r in rankings.items()}
    (out/'retrieval.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    dump(out/'metrics.json',summaries)
    config=dict(bm25=best_bm,rrf=best_rrf,graph=best_graph,protocol_sha256=sha(out/'protocol.json'),
                qa_dev_sha256=sha(qa_path),graph_sha256=sha(graph_path),selection=protocol['selection'],
                inference='All seeds/linking/fusion use question and retrieved papers only. Question type and gold IDs only used for DEV metrics/selection.',
                test_evaluated=False,semantic_graph='MENTIONS control; not a USES/PROPOSES semantic KG.')
    dump(out/'selected_config.json',config)
    dump(out/'runtime.json',dict(elapsed_seconds=time.perf_counter()-start,query_encoding_seconds=encoding_seconds,
        graph_stage_latency={k:dict(p50_ms=float(np.percentile(v,50)*1000),p95_ms=float(np.percentile(v,95)*1000)) for k,v in timings.items()},
        latency_scope='Warm graph stage only, excludes text/dense retrieval, linking and generation.',
        dense=dict(model=model,document_cache_sha256=sha(doc_path),
                   current_local_model_snapshot='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a',
                   historical_document_embedding_revision='Not recorded in original cache; ID/model key, dimensions, norms and cache SHA verified. Exact original encoder/text provenance not independently established.'),
        network='Offline; no API calls'))
    print(json.dumps({m:summaries[m] for m in [best_bm,'dense',best_rrf,'linked_text_expansion',best_graph]},indent=2),flush=True)
if __name__=='__main__':main()
