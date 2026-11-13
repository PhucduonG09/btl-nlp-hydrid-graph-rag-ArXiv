"""DEV stage diagnostics and oracle-A upper-bound probe, separate from selection."""
import csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kg.mentions import MentionIndex
from src.kg.aliases import match
from src.retrieval.graph import GraphSearch,fuse_rrf
from src.eval.benchmark_metrics import summarize
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def main():
    base=ROOT/'results/benchmark_v2_assisted/dev_01';qa=load(ROOT/'data/eval/benchmark_v2_assisted/qa_dev.jsonl')
    byid={q['id']:q for q in qa};runs=load(base/'retrieval.jsonl');cfg=json.loads((base/'selected_config.json').read_text())
    g=json.loads((ROOT/'data/kg/reference_mentions_v2_assisted.json').read_text(encoding='utf-8'))
    search=GraphSearch(MentionIndex(g));errors=[];bridge_rows=[];oracle={};expected_paths=0;any_paths=0
    strategy,kind=cfg['graph'].split('_')[:2]
    for run in runs:
        q=byid[run['id']]
        for method,ranking in run['rankings'].items():
            missing=[pid for pid in q['gold_ids'] if pid not in ranking[:10]]
            if missing:errors.append(dict(id=q['id'],type=q['type'],method=method,missing_top10=' | '.join(missing),
                                          gold_ranks=' | '.join(str(ranking.index(pid)+1) if pid in ranking else '>100' for pid in q['gold_ids'])))
        if q['type']!='bridge':continue
        a=q['hop1_id'];b=q['hop2_id'];seeds=run['seeds'][strategy]
        raw,paths=search.ppr(seeds) if kind=='ppr' else search.expand(seeds,hops=int(kind[1:]))
        path=paths.get(b,[]);any_paths+=bool(path)
        exact=any(match(edge['entity'],q['bridge_entity'],loose=True) for edge in path);expected_paths+=exact
        for edge in path:
            assert edge['A'] in g['entity_papers'][edge['entity']] and edge['B'] in g['entity_papers'][edge['entity']]
        ranking=run['rankings'][cfg['graph']];complete=set(q['gold_ids'])<=set(ranking[:10])
        if complete:cause='COMPLETE_TOP10'
        elif a not in seeds:cause='A_NOT_IN_SEEDS'
        elif not (set(g['paper_entities'][a])&set(g['paper_entities'][b])):cause='NO_SHARED_MENTION'
        elif b not in raw:cause='B_MISSING_FROM_GRAPH_CANDIDATES'
        elif b not in ranking[:10]:cause='B_CANDIDATE_RANKED_BELOW10'
        else:cause='A_RANKED_BELOW10'
        bridge_rows.append(dict(id=q['id'],A_in_seeds=a in seeds,B_in_graph_candidates=b in raw,complete_top10=complete,
                                automatic_cause=cause,path_to_B=path,path_matches_expected_entity=exact))
        oracle_graph,_=search.expand([a],hops=1)
        oracle[q['id']]=fuse_rrf([run['rankings'][cfg['rrf']],oracle_graph],weights=[1,2])
    with (base/'error_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['id','type','method','missing_top10','gold_ranks']);writer.writeheader();writer.writerows(errors)
    (base/'bridge_path_audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in bridge_rows),encoding='utf-8')
    summary=dict(failed_query_method_pairs=len(errors),unique_queries_with_any_failure=len({r['id'] for r in errors}),
                 selected_graph=cfg['graph'],selected_graph_bridge_causes=dict(Counter(r['automatic_cause'] for r in bridge_rows)),
                 bridge_queries=len(bridge_rows),B_with_recorded_graph_path=any_paths,recorded_path_matching_expected_entity=expected_paths,
                 oracle_A_only_diagnostic=summarize([q for q in qa if q['type']=='bridge'],oracle),
                 oracle_scope='Gold A seed on DEV, fixed one-hop and weight2 diagnostic. Not an inference system, not selected/tuned using these oracle results.',
                 caution='A MENTIONS path is a literal shared-name path, not proof of USES/PROPOSES or correctness. Automatic causes are not independent human error annotation. Repeated methods on one query are not independent cases.')
    (base/'error_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
