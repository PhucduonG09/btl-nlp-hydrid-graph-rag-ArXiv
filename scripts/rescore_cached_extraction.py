"""New versioned extraction score; no APIs, annotation edits or legacy overwrites."""
import argparse,csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.extraction_matching import relation_counts,prf

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--include-rule',action='store_true');args=ap.parse_args()
    gold_path=ROOT/'data/eval/gold_v4.csv';gold=defaultdict(list)
    for row in csv.DictReader(gold_path.open(encoding='utf-8-sig')):
        gold[row['paper_id']]
        if row['relation']!='NONE':gold[row['paper_id']].append(row)
    names=['gold_v4_v3.jsonl','gold_v4_v3_b4.jsonl','gold_v4_v3s_b1.jsonl','gold_v4_v3s_b4.jsonl']
    if args.include_rule:names.append('corpus_v2_rule_v1.jsonl')
    report=dict(gold_sha256=sha(gold_path),n_gold_papers=len(gold),n_gold_relations=sum(map(len,gold.values())),
                protocol='One-to-one maximum cardinality, canonical name/neutral suffix/pairwise acronym match. No evidence-overlap fallback. All gold tags included. Relation instances, not deduplicated entity sets. Pairwise acronym matching can remain ambiguous; inspect affected names.',
                human_provenance='Existing gold file; independent authorship is not established by this scoring run.',
                interpretation='Four cached configurations share a model family. Optional local-context rule baseline is a separate algorithm with narrower capabilities. No direct comparison to legacy many-to-many scores.',runs={})
    for name in names:
        path=ROOT/'data/kg'/name;records=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        pred={r['paper_id']:r.get('relations',[]) for r in records}
        assert len(pred)==len(records) and set(gold)<=set(pred)
        per_paper={}
        for typed in [False,True]:
            key='typed' if typed else 'entity_agnostic_instances'
            counts=[relation_counts(pred[pid],rels,typed=typed) for pid,rels in gold.items()]
            total={c:sum(row[c] for row in counts) for c in ['tp','fp','fn']}
            rng=np.random.default_rng(42);matrix=np.array([[c['tp'],c['fp'],c['fn']] for c in counts])
            samples=matrix[rng.integers(0,len(counts),size=(2000,len(counts)))].sum(axis=1)
            f1s=[prf(dict(zip(['tp','fp','fn'],s.tolist())))['F1'] for s in samples]
            per_paper[key]=dict(**prf(total),F1_paper_bootstrap_95=np.percentile(f1s,[2.5,97.5]).tolist())
        report['runs'][name]=dict(pred_sha256=sha(path),served_labels=dict(Counter(r.get('usage',{}).get('served_model',r.get('extractor','unknown')) for r in records)),scores=per_paper)
    out=ROOT/'results/analysis'/('extraction_one_to_one_v2.json' if args.include_rule else 'extraction_one_to_one_v1.json');out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({n:r['scores']['typed'] for n,r in report['runs'].items()},indent=2))
if __name__=='__main__':main()
