"""Apply explicitly accepted replacement proposals without altering source QA."""
import hashlib
from copy import deepcopy
from src.eval.benchmark_review import fingerprint
from src.kg.evidence import locate_evidence

def apply_replacements(rows,replacements,corpus):
    output=deepcopy(rows);by_id={r['id']:r for r in output}
    assert len(by_id)==len(rows)
    seen=set()
    for change in replacements:
        qid=change['id'];assert qid not in seen;seen.add(qid)
        row=by_id[qid]
        assert row['type']=='single' and row.get('origin')!='bridge_v1_approved','Bridge replacements are prohibited'
        assert fingerprint(row)==change['superseded_fingerprint'],'Replacement targets a changed source draft'
        proposal=change['replacement'];pid=proposal['paper_id'];paper=corpus[pid]
        assert not any(pid in other['gold_ids'] for other in output if other['id']!=qid),'Replacement paper already used'
        source=paper['title']+'. '+paper['abstract'];spans=[]
        assert proposal['title']==paper['title'] and proposal['abstract']==paper['abstract']
        for quote in proposal['evidence']:
            span=locate_evidence(source,quote);assert span,(qid,'Missing replacement evidence')
            spans.append(dict(text=span.text,start=span.start,end=span.end,match_kind=span.match_kind))
        row['superseded']=dict(question=row['question'],reference_answer=row['reference_answer'],gold_ids=row['gold_ids'],origin=row['origin'])
        row.update(question=proposal['question'],reference_answer=proposal['reference_answer'],gold_ids=[pid],
                   origin='new_authored_draft',review_status='pending',evidence_spans=spans,
                   source_text_sha256=hashlib.sha256(source.encode()).hexdigest(),numeric_review_hints=[],
                   replacement_approval=change['approval'],revision_reason='Hill đã chọn thay câu này bằng phương án có paper và bằng chứng mới.')
    return output
