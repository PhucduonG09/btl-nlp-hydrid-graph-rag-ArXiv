"""Corpus-only rule baseline; no QA/gold files or API calls."""
import hashlib,json,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kg.rule_extract import extract_one,VERSION
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'data/processed/benchmark_v2_corpus.jsonl';out=ROOT/'data/kg/corpus_v2_rule_v1.jsonl'
    papers=[json.loads(l) for l in source.read_text(encoding='utf-8').splitlines() if l.strip()]
    start=time.perf_counter();counts=Counter();nonempty=0
    with out.open('w',encoding='utf-8') as f:
        for p in papers:
            row=extract_one(p);text=p['title']+'. '+p['abstract']
            for r in row['relations']:
                assert text[r['evidence_start']:r['evidence_end']]==r['evidence']
                counts[r['relation']]+=1
            nonempty+=bool(row['relations']);f.write(json.dumps(row,ensure_ascii=False)+'\n')
    report=dict(extractor=VERSION,corpus_sha256=sha(source),output_sha256=sha(out),n_papers=len(papers),n_with_relations=nonempty,
                relations=dict(counts),elapsed_seconds=time.perf_counter()-start,
                code_sha256={p:sha(ROOT/p) for p in ['src/kg/rule_extract.py','src/kg/mentions.py','src/kg/aliases.py']},
                scope='Corpus-only weak rule baseline; independent of QA/gold. Not a replacement for validated semantic extraction. No ADDRESSES or unnamed concepts.')
    out.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
