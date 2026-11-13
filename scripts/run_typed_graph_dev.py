"""Fixed-parameter typed-rule ablation; no gold-conditioned graph construction."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kg.typed_reference import build_typed_graph,TypedGraphSearch
from src.kg.mentions import MentionIndex
from src.retrieval.graph import fuse_rrf
from src.eval.benchmark_metrics import summarize
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    corpus=ROOT/'data/processed/benchmark_v2_corpus.jsonl';extract=ROOT/'data/kg/corpus_v2_rule_v1.jsonl'
    extraction_manifest=json.loads(extract.with_suffix('.manifest.json').read_text())
    assert sha(corpus)==extraction_manifest['corpus_sha256'] and sha(extract)==extraction_manifest['output_sha256']
    mention_path=ROOT/'data/kg/reference_mentions_v2_assisted.json';mentions=json.loads(mention_path.read_text(encoding='utf-8'))
    assert mentions['corpus_sha256']==sha(corpus)
    graph=build_typed_graph(load(corpus),load(extract),mentions)
    graph.update(corpus_sha256=sha(corpus),extraction_sha256=sha(extract),normalization_mentions_sha256=sha(mention_path),code_sha256=sha(ROOT/'src/kg/typed_reference.py'))
    graph_path=ROOT/'data/kg/reference_typed_rule_v1.json';graph_path.write_text(json.dumps(graph,ensure_ascii=False),encoding='utf-8')
    search=TypedGraphSearch(MentionIndex(graph));base=ROOT/'results/benchmark_v2_assisted/dev_01'
    qa=load(ROOT/'data/eval/benchmark_v2_assisted/qa_dev.jsonl');cfg=json.loads((base/'selected_config.json').read_text())
    source=load(base/'retrieval.jsonl');rankings={name:{} for name in ['rrf','mentions_graph','typed_any_k1','typed_strict_k1']};records=[]
    for record in source:
        qid=record['id'];seeds=record['seeds']['linked'];rrf=record['rankings'][cfg['rrf']]
        rankings['rrf'][qid]=rrf;rankings['mentions_graph'][qid]=record['rankings'][cfg['graph']]
        item=dict(id=qid,seeds=seeds,paths={})
        for name,fn in [('typed_any_k1',lambda:search.expand(seeds,hops=1)),('typed_strict_k1',lambda:search.strict_onehop(seeds))]:
            raw,paths=fn();rankings[name][qid]=fuse_rrf([rrf,raw],k=60,weights=[1,2])
            item['paths'][name]={pid:paths[pid] for pid in raw[:20] if pid in paths}
        item['rankings']={m:r[qid] for m,r in rankings.items()};records.append(item)
    out=base/'typed_rule';out.mkdir(exist_ok=True)
    metrics={m:summarize(qa,r) for m,r in rankings.items()}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (out/'retrieval.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    (out/'protocol.json').write_text(json.dumps(dict(graph_sha256=sha(graph_path),seeds='Same query-linked seeds as the primary MENTIONS run; no gold seeds.',
        parameters='Fixed one-hop, seed_count3, final RRF k60 weights1:2; inherited from MENTIONS DEV choice, not independently tuned typed optimum.',
        strict='Source USES/EVALUATES_ON and destination PROPOSES only. Reference/baseline links outside this schema may be omitted.',test_access='None; DEV only.'),indent=2),encoding='utf-8')
    print(json.dumps(dict(graph=graph['stats'],bridge_all10={k:v['bridge']['All@10'] for k,v in metrics.items()}),indent=2))
if __name__=='__main__':main()
