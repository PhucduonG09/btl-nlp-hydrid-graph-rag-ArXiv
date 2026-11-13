"""Offline baseline evaluation on a finalized bridge QA version.

Reuses the existing pilot+distractor document embeddings after verifying their
ID/model cache key. Query embeddings are computed locally, with network access
disabled. Does not overwrite the midterm results or any workbook.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_baselines import PREFIX,rrf,tok
from src.eval.ir_metrics import METRICS,hit_all_at_k

def load(path):
    with path.open(encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def bm25_rank(papers,questions):
    """BM25Okapi k1=1.5, b=.75, negative-IDF floor .25*mean(IDF)."""
    postings=defaultdict(list);lengths=[]
    for i,p in enumerate(papers):
        tokens=tok(f"{p['title']}. {p['abstract']}");lengths.append(len(tokens))
        for term,freq in Counter(tokens).items():postings[term].append((i,freq))
    n=len(papers);lengths=np.asarray(lengths);avg=lengths.mean()
    idfs={term:np.log(n-len(v)+.5)-np.log(len(v)+.5) for term,v in postings.items()}
    floor=.25*sum(idfs.values())/len(idfs)
    needed={t for q in questions for t in tok(q)};weights={}
    for term in needed & postings.keys():
        pairs=np.asarray(postings[term]);doc,freq=pairs[:,0],pairs[:,1]
        weight=(idfs[term] if idfs[term]>=0 else floor)*freq*2.5/(freq+1.5*(.25+.75*lengths[doc]/avg))
        weights[term]=(doc,weight)
    ids=[p['paper_id'] for p in papers];ranked=[];timings=[]
    for q in questions:
        start=time.perf_counter();scores=np.zeros(n)
        for term in tok(q):
            if term in weights:
                doc,weight=weights[term];scores[doc]+=weight
        ranked.append([ids[i] for i in np.argsort(-scores)[:100]])
        timings.append(time.perf_counter()-start)
    return ranked,timings

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',default='v1');ap.add_argument('--bm25-only',action='store_true')
    args=ap.parse_args()
    base=ROOT/'data/eval'/f'bridge_{args.version}';manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    qa=load(base/'qa_all.jsonl');corpus_path=ROOT/manifest['corpus'];papers=load(corpus_path)
    assert sha(corpus_path)==manifest['corpus_sha256'] and sha(base/'qa_all.jsonl')==manifest['qa_sha256']
    out=ROOT/'results'/f'bridge_{args.version}';out.mkdir(parents=True,exist_ok=True)
    ids=[p['paper_id'] for p in papers];assert len(ids)==len(set(ids))
    assert all(set(q['gold_ids'])<=set(ids) for q in qa)
    queries=[q['question'] for q in qa];start=time.perf_counter()
    bm25,timings=bm25_rank(papers,queries);rank={'bm25':bm25}
    details=dict(bm25=dict(k1=1.5,b=.75,epsilon=.25,query_rank_seconds=timings))
    print(f'BM25 complete: {len(qa)} questions on {len(ids)} papers',flush=True)
    if not args.bm25_only:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(min(4,os.cpu_count() or 1))
        model_name='BAAI/bge-base-en-v1.5'
        source_papers=load(ROOT/'data/processed/pilot_papers.jsonl')+load(ROOT/'data/processed/distractors_20000.jsonl')
        source_ids=[p['paper_id'] for p in source_papers]
        key=hashlib.sha1(('|'.join(source_ids)+model_name).encode()).hexdigest()[:10]
        document_cache=ROOT/'data/index'/f'emb_scale_{key}.npy'
        assert document_cache.exists(),f'Missing verified document embedding cache {document_cache}'
        D_source=np.load(document_cache,mmap_mode='r');assert D_source.shape==(len(source_ids),768)
        lookup={pid:i for i,pid in enumerate(source_ids)}
        assert len(lookup)==len(source_ids)
        D=np.asarray(D_source[[lookup[pid] for pid in ids]],dtype='float32')
        assert np.isfinite(D).all() and np.allclose(np.linalg.norm(D,axis=1),1,atol=1e-3)
        print('Loading cached local BGE model for query encoding...',flush=True)
        encoder=SentenceTransformer(model_name,device='cpu',local_files_only=True)
        query_key=hashlib.sha256((model_name+'\n'+PREFIX+'\n'+'\n'.join(queries)).encode()).hexdigest()[:16]
        query_cache=out/f'query_embeddings_{query_key}.npy'
        encoding_start=time.perf_counter()
        if query_cache.exists():Q=np.load(query_cache)
        else:
            Q=encoder.encode([PREFIX+q for q in queries],normalize_embeddings=True,batch_size=8,show_progress_bar=True).astype('float32')
            np.save(query_cache,Q)
        assert Q.shape==(len(qa),D.shape[1]) and np.isfinite(Q).all()
        assert np.allclose(np.linalg.norm(Q,axis=1),1,atol=1e-3)
        S=Q@D.T
        rank['dense']=[[ids[i] for i in np.argsort(-scores)[:100]] for scores in S]
        rank['rrf']=[rrf([bm25[i],rank['dense'][i]],k=60,top=100) for i in range(len(qa))]
        details['dense']=dict(model=model_name,prefix=PREFIX,device='cpu',document_embedding_cache=str(document_cache.relative_to(ROOT)),
                              document_cache_sha256=sha(document_cache),query_encoding_seconds=time.perf_counter()-encoding_start,
                              document_embedding_revision='Not recorded by the original cached run; reused by verified ID/model key.')
        details['rrf']=dict(k=60,depth=100)
        print('Dense and RRF complete',flush=True)
    test_ids={q['id'] for q in load(base/'qa_test.jsonl')};dev_ids={q['id'] for q in load(base/'qa_dev.jsonl')}
    summary={};per_query=[]
    for method,lists in rank.items():
        with (out/f'retrieval_{method}.jsonl').open('w',encoding='utf-8') as f:
            for q,rl in zip(qa,lists):f.write(json.dumps(dict(id=q['id'],ranked=rl),ensure_ascii=False)+'\n')
        for split,include in [('all',{q['id'] for q in qa}),('dev',dev_ids),('test',test_ids)]:
            indexes=[i for i,q in enumerate(qa) if q['id'] in include]
            metrics={name:round(float(np.mean([fn(lists[i],qa[i]['gold_ids']) for i in indexes])),4) for name,fn in METRICS.items()}
            for k in [5,10]:
                metrics[f'All@{k}']=round(float(np.mean([hit_all_at_k(lists[i],qa[i]['gold_ids'],k) for i in indexes])),4)
                for hop in [1,2]:metrics[f'Hop{hop}@{k}']=round(float(np.mean([qa[i][f'hop{hop}_id'] in lists[i][:k] for i in indexes])),4)
            summary[f'{split}/{method}']=dict(n=len(indexes),**metrics)
        for q,rl in zip(qa,lists):
            per_query.append(dict(id=q['id'],method=method,split='test' if q['id'] in test_ids else 'dev',
                                  A_rank=rl.index(q['hop1_id'])+1 if q['hop1_id'] in rl else None,
                                  B_rank=rl.index(q['hop2_id'])+1 if q['hop2_id'] in rl else None,
                                  all_at_10=bool(hit_all_at_k(rl,q['gold_ids'],10))))
    (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'per_query.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in per_query),encoding='utf-8')
    (out/'run_manifest.json').write_text(json.dumps(dict(dataset_qa_sha256=manifest['qa_sha256'],corpus_sha256=manifest['corpus_sha256'],
                              elapsed_seconds=time.perf_counter()-start,systems=details,network='offline',
                              caveat=manifest['interpretation']),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k.startswith('all/')},indent=2),flush=True)

if __name__=='__main__':main()
