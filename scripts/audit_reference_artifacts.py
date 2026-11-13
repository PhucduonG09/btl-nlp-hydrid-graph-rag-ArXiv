"""Verify immutable review files, frozen QA and corpus-only graph evidence."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def main():
    base=ROOT/'data/eval/bridge_v1';m=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    workbooks={name:sha(ROOT/'data/eval'/name)==digest for name,digest in m['source_workbook_sha256'].items()}
    assert all(workbooks.values()),'A source workbook changed since freeze'
    assert sha(base/'qa_all.jsonl')==m['qa_sha256']
    assert sha(ROOT/m['corpus'])==m['corpus_sha256']
    dev=load(base/'qa_dev.jsonl');test=load(base/'qa_test.jsonl');allq=load(base/'qa_all.jsonl')
    assert {q['id'] for q in dev}.isdisjoint(q['id'] for q in test)
    assert {p for q in dev for p in q['gold_ids']}.isdisjoint(p for q in test for p in q['gold_ids'])
    assert {q['id'] for q in dev+test}=={q['id'] for q in allq}
    gpath=ROOT/'data/kg/reference_mentions_v1.json';g=json.loads(gpath.read_text(encoding='utf-8'))
    assert g['corpus_sha256']==m['corpus_sha256']
    assert g['build_source_sha256']==sha(ROOT/'src/kg/mentions.py')
    papers={p['paper_id']:p for p in load(ROOT/m['corpus'])};count=0
    assert set(g['paper_ids'])==set(papers)
    for pid,entities in g['paper_entities'].items():
        p=papers[pid];source=p['title']+'. '+p['abstract']
        for entity in entities:
            span=g['evidence'][pid][entity]
            assert 0<=span['start']<span['end']<=len(source)
            assert source[span['start']:span['end']].strip()
            assert pid in g['entity_papers'][entity]
            count+=1
    assert count==g['stats']['mention_edges']
    report=dict(ok=True,source_workbooks_unchanged=workbooks,qa_count=len(allq),dev_count=len(dev),test_count=len(test),
                no_shared_gold_papers=True,verified_mention_spans=count,graph_sha256=sha(gpath),
                test_access='Integrity-only checks of IDs and split membership; no test retrieval or tuning.')
    out=ROOT/'results/bridge_v1/reference_artifact_audit.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
