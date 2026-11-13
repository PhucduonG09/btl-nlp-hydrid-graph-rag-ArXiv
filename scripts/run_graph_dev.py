"""DEV-only ablation; corpus-only graph, query-only linking, no gold seeding."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kg.mentions import MentionIndex
from src.retrieval.graph import GraphSearch,fuse_rrf,fuse_weighted
from src.eval.ir_metrics import METRICS
from scripts.run_bridge_baselines import bm25_rank

def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    graph_path=ROOT/'data/kg/reference_mentions_v1.json'
    data=json.loads(graph_path.read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'data/eval/bridge_v1/manifest.json').read_text(encoding='utf-8'))
    assert data['corpus_sha256']==manifest['corpus_sha256']
    assert sha(ROOT/manifest['corpus'])==manifest['corpus_sha256']
    assert data['build_source_sha256']==sha(ROOT/'src/kg/mentions.py'),'Rebuild graph after source changes'
    index=MentionIndex(data);search=GraphSearch(index)
    qa=load(ROOT/'data/eval/bridge_v1/qa_dev.jsonl')
    saved={m:{r['id']:r['ranked'] for r in load(ROOT/f'results/bridge_v1/retrieval_{m}.jsonl')} for m in ['bm25','dense','rrf']}
    out=ROOT/'results/bridge_v1/graph_dev';out.mkdir(exist_ok=True)
    records=[];elapsed={};rankings={m:[] for m in saved}
    papers={p['paper_id']:p for p in load(ROOT/manifest['corpus'])}
    expansion_queries=[]
    for q in qa:
        base=saved['rrf'][q['id']]
        for strategy in ['rrf','linked']:
            seeds=base[:3] if strategy=='rrf' else index.linked_seeds(q['question'],base,count=3)
            for pid in seeds:
                p=papers[pid];expansion_queries.append(q['question']+' '+p['title']+'. '+p['abstract'])
    lexical,_=bm25_rank(list(papers.values()),expansion_queries)
    lexical_cursor=0
    for q in qa:
        qid=q['id'];base=saved['rrf'][qid];rankings_q={m:saved[m][qid] for m in saved}
        paths_q={};seeds_q={}
        for strategy in ['rrf','linked']:
            seeds=base[:3] if strategy=='rrf' else index.linked_seeds(q['question'],base,count=3)
            seeds_q[strategy]=seeds
            rankings_q[strategy+'_text_expansion']=fuse_rrf([base]+lexical[lexical_cursor:lexical_cursor+len(seeds)])
            lexical_cursor+=len(seeds)
            for hop in [1,2,3]:
                start=time.perf_counter();ranked,paths=search.expand(seeds,hops=hop)
                name=f'{strategy}_k{hop}'
                for fusion in ['rrf','weighted']:
                    method=name+'_'+fusion
                    rankings_q[method]=fuse_rrf([base,ranked]) if fusion=='rrf' else fuse_weighted([base,ranked],[.5,.5])
                    paths_q[method]=paths
                elapsed.setdefault(name,[]).append(time.perf_counter()-start)
            start=time.perf_counter();ranked,paths=search.ppr(seeds)
            name=strategy+'_ppr_rrf';rankings_q[name]=fuse_rrf([base,ranked]);paths_q[name]=paths
            elapsed.setdefault(name,[]).append(time.perf_counter()-start)
        for method,ranking in rankings_q.items():
            if method not in rankings:rankings[method]=[]
            rankings[method].append(ranking)
        records.append(dict(id=qid,seeds=seeds_q,rankings=rankings_q,paths=paths_q))
        print(f'{len(records)}/{len(qa)}',flush=True)
    metrics={}
    for method,lists in rankings.items():
        scores={k:round(float(np.mean([fn(r,q['gold_ids']) for r,q in zip(lists,qa)])),6) for k,fn in METRICS.items()}
        scores['Hop2@10']=round(float(np.mean([q['hop2_id'] in r[:10] for r,q in zip(lists,qa)])),6)
        metrics[method]=scores
    candidates=[m for m in metrics if m not in saved and not m.endswith('_text_expansion')]
    best=sorted(candidates,key=lambda m:(-metrics[m]['All@10'],-metrics[m]['Hop2@10'],-metrics[m]['MRR@10'],m))[0]
    config=dict(selected=best,seed_count=3,beam=100,entity_limit=20,ppr_alpha=.85,ppr_iterations=60,rrf_k=60,
                fusion_depth=100,weighted_rank_weights=[.5,.5],graph_sha256=sha(graph_path),
                dev_sha256=sha(ROOT/'data/eval/bridge_v1/qa_dev.jsonl'),selection='DEV All@10, then Hop2@10, then MRR@10, then method name',
                scope='Exploratory BM25-screened bridge DEV only. Test not run. MENTIONS control, not semantic KG.')
    (out/'selected_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'retrieval.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf-8')
    latency={m:dict(p50_ms=float(np.percentile(v,50)*1000),p95_ms=float(np.percentile(v,95)*1000)) for m,v in elapsed.items()}
    (out/'latency.json').write_text(json.dumps(dict(scope='Warm graph-stage only; excludes graph load, BM25, dense encoding and generation. k-hop timing includes both fusion variants.',methods=latency),indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=best,baseline=metrics['rrf'],graph=metrics[best]),indent=2))
if __name__=='__main__':main()
