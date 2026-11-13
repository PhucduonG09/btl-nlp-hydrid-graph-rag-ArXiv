"""Evidence-backed DEV errors; automatic labels are hypotheses, not human ratings."""
import csv,json,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def load(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
def main():
    out=ROOT/'results/bridge_v1/graph_dev'
    qa={q['id']:q for q in load(ROOT/'data/eval/bridge_v1/qa_dev.jsonl')}
    g=json.loads((ROOT/'data/kg/reference_mentions_v1.json').read_text(encoding='utf-8'))
    rows=[]
    for run in load(out/'retrieval.jsonl'):
        q=qa[run['id']];a=q['hop1_id'];b=q['hop2_id']
        shared=set(g['paper_entities'][a])&set(g['paper_entities'][b])
        for method,ranked in run['rankings'].items():
            if set(q['gold_ids'])<=set(ranked[:10]):continue
            strategy='linked' if method.startswith('linked_') else 'rrf'
            seeds=run['seeds'][strategy]
            if method in {'bm25','dense','rrf'}:reason='BASELINE_TOP10_INCOMPLETE'
            elif method.endswith('_text_expansion'):reason='TEXT_EXPANSION_TOP10_INCOMPLETE'
            elif a not in seeds:reason='HOP1_NOT_IN_SEEDS'
            elif not shared:reason='NO_SHARED_MENTION_ENTITY'
            elif b not in ranked:reason='BRIDGE_EXISTS_B_NOT_IN_CANDIDATES'
            elif b not in ranked[:10]:reason='B_PRESENT_BUT_RANK_GT_10'
            else:reason='A_DROPPED_FROM_TOP10'
            rows.append(dict(id=q['id'],method=method,question=q['question'],hop1=a,hop2=b,
                             A_rank=ranked.index(a)+1 if a in ranked else '>100',B_rank=ranked.index(b)+1 if b in ranked else '>100',
                             shared_entities=' | '.join(sorted(shared)),seeds=' | '.join(seeds),automatic_hypothesis=reason,
                             human_verdict='',human_note=''))
    with (out/'errors_to_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(n_failed_query_method_pairs=len(rows),unique_failed_queries=len({r['id'] for r in rows}),
                 hypotheses=dict(Counter(r['automatic_hypothesis'] for r in rows)),
                 scope='DEV only. Repeated queries across methods are not independent error cases. Human columns intentionally blank.')
    (out/'error_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
