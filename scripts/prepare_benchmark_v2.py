"""Stage corpus and source-only candidate papers; never modify bridge v1 or Excel."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def main():
    out=ROOT/'data/eval/benchmark_v2_draft';out.mkdir(exist_ok=True)
    sources=[ROOT/'data/processed/pilot_papers.jsonl',ROOT/'data/processed/distractors_20000.jsonl']
    corpus={}
    for source in sources:
        for p in load(source):
            if p['paper_id'] in corpus:
                assert all(p[k]==corpus[p['paper_id']][k] for k in ['title','abstract'])
            corpus[p['paper_id']]=p
    old=load(ROOT/'data/eval/qa.jsonl');single=[q for q in old if q['type']=='single']
    bridge=load(ROOT/'data/eval/bridge_v1/qa_all.jsonl')
    assert len(single)==60 and len(bridge)==45
    assert all(set(q['gold_ids'])<=set(corpus) for q in old+bridge)
    occupied={pid for q in old+bridge for pid in q['gold_ids']}
    candidates=[p for p in corpus.values() if p['paper_id'] not in occupied and 'cs.CL' in p.get('categories',[]) and 500<=len(p['abstract'])<=1900]
    candidates.sort(key=lambda p:hashlib.sha256(('single-v2-seed42:'+p['paper_id']).encode()).hexdigest())
    candidates=candidates[:45];assert len(candidates)==45
    selected=[dict(id=f's2{i:03d}',paper_id=p['paper_id'],title=p['title'],abstract=p['abstract'],categories=p['categories'],
                   source_text_sha256=hashlib.sha256((p['title']+'. '+p['abstract']).encode()).hexdigest()) for i,p in enumerate(candidates,1)]
    target=out/'new_single_sources.jsonl'
    if target.exists():assert load(target)==selected,'Candidate source selection changed; create a new version'
    else:write(target,selected)
    corpus_path=ROOT/'data/processed/benchmark_v2_corpus.jsonl';write(corpus_path,list(corpus.values()))
    write(out/'legacy_single_sources.jsonl',[dict(**q,papers=[corpus[p] for p in q['gold_ids']]) for q in single])
    write(out/'legacy_comparisons_separate.jsonl',[q for q in old if q['type']!='single'])
    manifest=dict(status='DRAFT_NOT_FROZEN_NOT_FINAL_TEST',n_corpus=len(corpus),n_legacy_single=len(single),n_approved_bridge=len(bridge),n_new_single_sources=len(selected),
                  planned_total=150,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources+[ROOT/'data/eval/qa.jsonl',ROOT/'data/eval/bridge_v1/qa_all.jsonl']},
                  corpus=str(corpus_path.relative_to(ROOT)),corpus_sha256=sha(corpus_path),
                  candidate_policy='SHA256 order with fixed seed42; cs.CL, abstract length 500..1900, exclude all existing QA gold papers. No KG, retrieval score or test performance used. This is a scoped convenience sample, not a representative natural-query distribution.',
                  split_policy='Do not freeze until QA review. Retain bridge-v1 DEV/test assignments; legacy QA have historical exposure and must be labeled accordingly, not described as a fully unseen test.')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['status','n_corpus','n_legacy_single','n_approved_bridge','n_new_single_sources','planned_total']},indent=2))
if __name__=='__main__':main()
