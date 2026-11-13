"""Receipt for Hill's explicit chat instruction: 'thay cả 3'."""
import hashlib,json,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.benchmark_review import fingerprint,save,decisions,effective
from scripts.compile_benchmark_v2_draft import main as compile_draft
BASE=ROOT/'data/eval/benchmark_v2_draft'
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    receipt=BASE/'accepted_replacements.json';cases=json.loads((BASE/'hill_decisions_needed.json').read_text(encoding='utf-8'))
    assert {c['id'] for c in cases}=={'s2006','s2019','q056'}
    if not receipt.exists():
        rows={r['id']:r for r in load(BASE/'qa_candidates.jsonl')}
        audit=json.loads((BASE/'content_review_v1.json').read_text(encoding='utf-8'))
        assert all(fingerprint(rows[r['id']])==r['source_fingerprint'] for r in audit['rows'])
        approvals=[]
        for case in cases:
            approvals.append(dict(id=case['id'],superseded_fingerprint=fingerprint(rows[case['id']]),replacement=case['replacement'],
                                  approval=dict(reviewer='Hill',channel='chat',instruction='thay cả 3',
                                                scope='Accept these three prepared replacements only; not independent approval of all QA.',
                                                recorded_at=datetime.now(timezone(timedelta(hours=7))).isoformat())))
        dump(receipt,approvals)
    changes=json.loads(receipt.read_text(encoding='utf-8'));compile_draft()
    rows={r['id']:r for r in load(BASE/'qa_candidates.jsonl')}
    stored=decisions(BASE/'review_decisions.sqlite3')
    written=0
    for change in changes:
        row=rows[change['id']];proposal=change['replacement']
        assert row['gold_ids']==[proposal['paper_id']] and row['question']==proposal['question']
        prior=stored.get(row['id'])
        if not prior or prior.get('source_sha256')!=fingerprint(row):
            save(BASE/'review_decisions.sqlite3',row,question=row['question'],answer=row['reference_answer'],status='approved',reviewer='Hill',
                 note='Chấp nhận phương án thay đã trình bày, theo chỉ dẫn chat “thay cả 3”. Chỉ áp dụng cho ba câu này.')
            written+=1
    audit=json.loads((BASE/'content_review_v1.json').read_text(encoding='utf-8'))
    for item in audit['rows']:
        if item['id'] in {c['id'] for c in changes}:
            item.update(status='cleared',source_fingerprint=fingerprint(rows[item['id']]),
                        note='Đã thay theo chỉ dẫn chat của Hill; paper, đáp án và bằng chứng được cập nhật cùng nhau.',resolution_reviewer='Hill')
        else:assert item['source_fingerprint']==fingerprint(rows[item['id']])
    audit.update(cleared_single=105,needs_hill=[],resolved_ids=[c['id'] for c in changes],
                 supersedes='content_review_v1.json',reviewer_kind='assistant_source_review_with_three_explicit_user_resolutions')
    dump(BASE/'content_review_v2.json',audit)
    for case in cases:case.update(applied=True,resolution='replace',approved_by='Hill')
    dump(BASE/'hill_decisions_needed.json',cases)
    print(json.dumps(dict(total=len(rows),content_pending=0,replacements={c['id']:c['replacement']['paper_id'] for c in changes},
                          unchanged_single_reviews=102,new_human_approval_records=written),indent=2))
if __name__=='__main__':main()
