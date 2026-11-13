"""Audit by default; freeze only with explicit --freeze and complete reviews."""
import argparse,hashlib,json,sys
from collections import Counter
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.benchmark_review import decisions,effective,curated_rows
from src.eval.benchmark_split import split_rows
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def write(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--freeze',action='store_true',help='Strict user-recorded review mode')
    mode.add_argument('--freeze-assisted',action='store_true',help='Separate experimental version with assistant-source-review provenance')
    args=ap.parse_args()
    base=ROOT/'data/eval/benchmark_v2_draft';manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    validation=json.loads((base/'validation.json').read_text(encoding='utf-8'))
    path=base/'qa_candidates.jsonl'
    assert sha(path)==validation['qa_candidates_sha256'],'Recompile after changing drafts'
    assert sha(ROOT/manifest['corpus'])==manifest['corpus_sha256'],'Corpus changed'
    for source,digest in manifest['source_sha256'].items():assert sha(ROOT/source)==digest,source
    source_rows=load(path);stored=decisions(base/'review_decisions.sqlite3');rows=[effective(r,stored) for r in source_rows]
    audit_path=base/'content_review_v2.json'
    audit={r['id']:r for r in json.loads(audit_path.read_text(encoding='utf-8'))['rows']}
    strict=curated_rows(rows,audit,allow_assisted=False)
    assisted=curated_rows(rows,audit,allow_assisted=True)
    accepted=assisted if args.freeze_assisted else strict
    pending=[r['id'] for r in rows if r['review_status'] not in {'approved','approved_existing','rejected'}]
    rejected=[r['id'] for r in rows if r['review_status']=='rejected']
    status=dict(candidates=len(rows),accepted=len(accepted),pending=len(pending),rejected=len(rejected),ready=len(accepted)>=150 and not pending and not rejected)
    status['user_recorded_or_prior_bridge']=len(strict)
    status['source_curated']=len(assisted)
    status['source_curation_pending']=len(rows)-len(assisted)
    if args.freeze_assisted:status['ready']=len(assisted)>=150 and len(assisted)==len(rows) and not rejected
    print(json.dumps(status,indent=2))
    if not (args.freeze or args.freeze_assisted):return
    if not status['ready']:raise SystemExit('Not frozen: incomplete reviews for the selected review mode.')
    version='v2_assisted' if args.freeze_assisted else 'v2'
    target=ROOT/'data/eval'/('benchmark_'+version)
    if target.exists():raise SystemExit('Frozen output exists. Create a new version instead of overwriting it.')
    fixed={q['id']:part for part in ['dev','test'] for q in load(ROOT/f'data/eval/bridge_v1/qa_{part}.jsonl')}
    split=split_rows(accepted,fixed)
    ids=[q['id'] for q in accepted];assert len(ids)==len(set(ids))
    assert len({q['question'].strip().casefold() for q in accepted})==len(accepted),'Duplicate questions'
    corpus_ids={p['paper_id'] for p in load(ROOT/manifest['corpus'])}
    assert all(set(q['gold_ids'])<=corpus_ids for q in accepted)
    bridge_original={q['id']:q for q in load(ROOT/'data/eval/bridge_v1/qa_all.jsonl')}
    for q in accepted:
        if q['id'] in bridge_original:
            assert all(q[k]==bridge_original[q['id']][k] for k in ['question','reference_answer','gold_ids'])
    target.mkdir()
    write(target/'qa_all.jsonl',accepted)
    for part,items in split.items():write(target/f'qa_{part}.jsonl',items)
    result=dict(version=version,created_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),seed=42,
                n_qa=len(accepted),n_dev=len(split['dev']),n_test=len(split['test']),n_papers=len(corpus_ids),
                corpus=manifest['corpus'],corpus_sha256=manifest['corpus_sha256'],source_candidate_sha256=sha(path),
                review_db_sha256=sha(base/'review_decisions.sqlite3'),
                source_review_sha256=sha(audit_path),
                curation_counts=dict(Counter(q['curation_provenance'] for q in accepted)),
                all_independently_human_annotated=False,
                review_mode='assisted_source_review_experimental' if args.freeze_assisted else 'user_recorded_review',
                qa_sha256={p.name:sha(p) for p in target.glob('*.jsonl')},
                split_origin_counts={part:dict(Counter(q['origin'] for q in items)) for part,items in split.items()},
                split_policy='Connected components by gold paper; preserve bridge-v1 split. Allocate 30% of unanchored new-paper components to test with seed42. Other legacy components to DEV.',
                exposure='Legacy questions have prior experimental exposure, including any forced into test by a bridge anchor. Report results by origin; do not describe all v2 test questions as previously unseen.',
                limitations='Bridge subset BM25-screened. New single papers are a scoped cs.CL convenience sample. Assistant review is not independent human labeling. This version does not satisfy the human-annotation A+ criterion by itself. Approval records do not establish independent double annotation.')
    (target/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen '+str(target))
if __name__=='__main__':main()
