"""Read approved workbook rows, validate them, and freeze a paper-disjoint QA split.

This script never saves or changes an Excel workbook.
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[('bridge_drafts2.xlsx','bn'),('bridge_drafts.xlsx','bd'),('bridge_handwritten.xlsx','bh')]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(rows):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--version',default='v1')
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if not re.fullmatch(r'v[0-9]+',args.version):raise ValueError('Version must be v1, v2, ...')
    out=ROOT/'data/eval'/f'bridge_{args.version}'
    qa=[];source_hashes={};counts={}
    for filename,prefix in SOURCES:
        path=ROOT/'data/eval'/filename
        source_hashes[filename]=sha(path)
        book=load_workbook(path,read_only=True,data_only=True)
        rows=book['Viết tay'].iter_rows(min_row=2,values_only=True)
        counts[filename]=Counter()
        for row in rows:
            n,term,A,B,status,question,answer,note=(list(row)+[None]*8)[:8]
            status=str(status or '').strip().upper()
            counts[filename][status or 'pending']+=1
            if status!='Y':continue
            assert question and answer,(filename,n,'missing QA')
            matches=[re.match(r'\[([^]]+)\]',str(p)) for p in (A,B)]
            assert all(matches),(filename,n,'invalid paper IDs')
            aid,bid=[m.group(1) for m in matches]
            assert aid!=bid,(filename,n,'identical hops')
            assert not re.search(r'(?<![A-Za-z0-9])'+re.escape(str(term))+r'(?![A-Za-z0-9])',str(question),re.I),(filename,n,'bridge name in question')
            qa.append(dict(id=f'{prefix}{int(n):03d}',type='bridge',question=str(question).strip(),reference_answer=str(answer).strip(),
                           gold_ids=[aid,bid],hop1_id=aid,hop2_id=bid,bridge_entity=str(term),group=bid,
                           source=filename,source_row=int(n)))
        book.close()
        assert source_hashes[filename]==sha(path),'Workbook changed while being read'
    assert len({q['id'] for q in qa})==len(qa),'Duplicate QA IDs'
    assert len({tuple(q['gold_ids']) for q in qa})==len(qa),'Duplicate paper pairs'
    assert len({q['question'].casefold() for q in qa})==len(qa),'Duplicate questions'

    corpus_sources=[ROOT/'data/processed/slice_c1_3000.jsonl',ROOT/'data/processed/distractors_20000.jsonl']
    papers={}
    for path in corpus_sources:
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                p=json.loads(line)
                if p['paper_id'] in papers:
                    previous=papers[p['paper_id']]
                    assert all(' '.join(previous[k].split())==' '.join(p[k].split()) for k in ('title','abstract')),(p['paper_id'],'conflicting corpus text')
                else:papers[p['paper_id']]=p
    missing=[(q['id'],pid) for q in qa for pid in q['gold_ids'] if pid not in papers]
    assert not missing,missing

    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for q in qa:parent[find(q['hop1_id'])]=find(q['hop2_id'])
    groups={}
    for q in qa:groups.setdefault(find(q['hop1_id']),[]).append(q)
    components=sorted(groups.values(),key=lambda qs:sorted(q['id'] for q in qs))
    random.Random(args.seed).shuffle(components)
    target=round(.3*len(qa));dev=[];test=[]
    for component in components:
        (test if len(test)+len(component)<=target else dev).extend(component)
    dev.sort(key=lambda q:q['id']);test.sort(key=lambda q:q['id'])
    dev_papers={p for q in dev for p in q['gold_ids']};test_papers={p for q in test for p in q['gold_ids']}
    assert not dev_papers&test_papers,'Gold papers overlap between dev and test'
    assert len(dev)+len(test)==len(qa) and dev and test

    all_text=jsonl(qa);qa_hash=hashlib.sha256(all_text.encode()).hexdigest()
    if (out/'manifest.json').exists():
        existing=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        assert existing.get('qa_content_sha256',existing['qa_sha256'])==qa_hash,'Existing version differs; choose a new version'
        assert sha(out/'qa_all.jsonl')==existing['qa_sha256'],'Frozen QA bytes differ from manifest'
        print(f'Already finalized {out.name}: {len(qa)} QA. No files changed.');return
    out.mkdir(parents=True,exist_ok=True)
    corpus_path=ROOT/'data/processed'/f'bridge_corpus_{args.version}.jsonl'
    corpus_path.write_text(jsonl(papers.values()),encoding='utf-8')
    for name,records in [('qa_all.jsonl',qa),('qa_dev.jsonl',dev),('qa_test.jsonl',test)]:
        (out/name).write_text(jsonl(records),encoding='utf-8')
    # Keep the working export synchronized; versioned snapshots are authoritative.
    (ROOT/'data/eval/qa_bridge_reviewed.jsonl').write_text(all_text,encoding='utf-8')
    manifest=dict(version=args.version,created_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),
                  n_qa=len(qa),n_dev=len(dev),n_test=len(test),n_papers=len(papers),n_hop2_groups=len({q['hop2_id'] for q in qa}),
                  n_paper_components=len(components),seed=args.seed,qa_sha256=sha(out/'qa_all.jsonl'),qa_content_sha256=qa_hash,corpus_sha256=sha(corpus_path),
                  corpus=str(corpus_path.relative_to(ROOT)),source_workbook_sha256=source_hashes,
                  corpus_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in corpus_sources},
                  workbook_status_counts={name:dict(c) for name,c in counts.items()},
                  split_policy='Connected components over both gold-paper IDs; no gold paper shared between dev and test.',
                  difficulty_screen='Questions were screened for BM25 B_rank > 10 on this corpus.',
                  interpretation='This is a BM25-screened bridge diagnostic set, not an unbiased sample of natural queries. A low BM25 score is a selection effect, not evidence of graph benefit.',
                  checks=dict(missing_gold=0,duplicate_pairs=0,duplicate_questions=0,gold_paper_overlap=0,workbooks_unchanged=True))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['version','n_qa','n_dev','n_test','n_papers','n_hop2_groups','n_paper_components','checks']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
