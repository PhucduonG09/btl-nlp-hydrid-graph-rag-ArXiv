"""Build a separate diagnostic KG only after all missing DEV papers are extracted."""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(path):
    with path.open(encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    old_graph=ROOT/'data/kg/kg_v3.graphml';old_hash=sha(old_graph)
    latest={r['paper_id']:r for r in load(ROOT/'data/kg/bridge_dev_v1_extract_v3.jsonl')}
    expected={p['paper_id'] for p in load(ROOT/'data/processed/bridge_v1_dev_missing_kg_papers.jsonl')}
    assert expected<=set(latest),'Some DEV papers have no extraction'
    assert all('error' not in latest[p] for p in expected),'Some DEV extractions are invalid'
    records={r['paper_id']:r for r in load(ROOT/'data/kg/pilot_extract_v3.jsonl')}
    assert not set(records)&expected,'Diagnostic missing-paper input overlaps the existing pilot'
    records.update({p:latest[p] for p in sorted(expected)})
    manifest=json.loads((ROOT/'data/eval/bridge_v1/manifest.json').read_text(encoding='utf-8'))
    papers={p['paper_id']:p for p in load(ROOT/'data/processed/pilot_papers.jsonl')+load(ROOT/manifest['corpus'])}
    assert set(records)<=set(papers)
    metadata=ROOT/'data/processed/bridge_dev_v1_diagnostic_papers.jsonl'
    merged=ROOT/'data/kg/bridge_dev_v1_diagnostic_extract.jsonl'
    metadata.write_text(''.join(json.dumps(papers[p],ensure_ascii=False)+'\n' for p in records),encoding='utf-8')
    merged.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records.values()),encoding='utf-8')
    tag='bridge_dev_v1_v3'
    subprocess.run([sys.executable,str(ROOT/'scripts/build_kg.py'),'--papers',str(metadata),'--extract',str(merged),'--tag',tag],check=True,cwd=ROOT)
    assert sha(old_graph)==old_hash,'Original KG unexpectedly changed'
    report=dict(tag=tag,n_papers=len(records),n_new_dev_papers=len(expected),
                served_models=dict(Counter(r.get('usage',{}).get('served_model','unknown') for r in records.values())),
                old_graph_sha256=old_hash,new_graph_sha256=sha(ROOT/'data/kg'/f'kg_{tag}.graphml'),
                warning='DEV gold-conditioned diagnostic graph. Existing pilot and new extractions can use different models. Not a fair final full-corpus Graph-RAG experiment.')
    (ROOT/'results/bridge_v1/diagnostic_kg_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
