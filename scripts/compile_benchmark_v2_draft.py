"""Validate authored evidence spans and assemble a reviewable, unfrozen benchmark."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kg.evidence import locate_evidence
from src.eval.benchmark_replacements import apply_replacements
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def write(p,rows):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def numbers(s):
    s=s.replace('\\%','%').replace('$','')
    return {str(float(x)) for x in re.findall(r'(?<![A-Za-z0-9])(?:\d+\.\d+|\.\d+|\d+)(?![A-Za-z0-9])',s)}
def main():
    base=ROOT/'data/eval/benchmark_v2_draft'
    sources={r['id']:r for r in load(base/'new_single_sources.jsonl')}
    authored=[]
    for path in sorted(base.glob('authored_single_*.json')):authored+=json.loads(path.read_text(encoding='utf-8'))
    assert len(authored)==45 and {r['id'] for r in authored}==set(sources)
    assert len({r['question'].casefold() for r in authored})==45
    new=[]
    for row in authored:
        source=sources[row['id']];text=source['title']+'. '+source['abstract'];spans=[]
        for quote in row['evidence']:
            span=locate_evidence(text,quote)
            assert span is not None,(row['id'],'evidence absent',quote)
            spans.append(dict(text=span.text,start=span.start,end=span.end,match_kind=span.match_kind))
        new.append(dict(id=row['id'],type='single',question=row['question'],reference_answer=row['reference_answer'],
                        gold_ids=[source['paper_id']],evidence_spans=spans,origin='new_authored_draft',review_status='pending',
                        numeric_review_hints=sorted(numbers(row['reference_answer'])-numbers(text)),
                        source_text_sha256=source['source_text_sha256']))
    legacy=load(base/'legacy_single_sources.jsonl')
    old=[{**{k:v for k,v in q.items() if k!='papers'},'origin':'legacy_single','review_status':'legacy_scope_check',
          'numeric_review_hints':sorted(numbers(q['reference_answer'])-numbers(' '.join(p['abstract'] for p in q['papers'])))} for q in legacy]
    revisions={r['id']:r for r in json.loads((base/'legacy_revision_proposals.json').read_text(encoding='utf-8'))}
    for row in old:
        prior=next(q for q in legacy if q['id']==row['id'])
        row['source_text_sha256']=hashlib.sha256('\n'.join(p['title']+'. '+p['abstract'] for p in prior['papers']).encode()).hexdigest()
        if row['id'] in revisions:
            proposal=revisions[row['id']]
            row['previous_question']=row['question'];row['previous_reference_answer']=row['reference_answer']
            for key in ['question','reference_answer']:
                if key in proposal:row[key]=proposal[key]
            row['revision_reason']=proposal['reason'];row['review_status']='revision_pending'
    bridge=[dict(**q,origin='bridge_v1_approved',review_status='approved_existing') for q in load(ROOT/'data/eval/bridge_v1/qa_all.jsonl')]
    combined=old+new+bridge
    replacements_path=base/'accepted_replacements.json'
    replacements=json.loads(replacements_path.read_text(encoding='utf-8')) if replacements_path.exists() else []
    if replacements:
        corpus={p['paper_id']:p for p in load(ROOT/'data/processed/benchmark_v2_corpus.jsonl')}
        combined=apply_replacements(combined,replacements,corpus)
    new=[r for r in combined if r['origin']=='new_authored_draft']
    old=[r for r in combined if r['origin']=='legacy_single']
    write(base/'new_single_drafts.jsonl',new)
    assert len(combined)==150 and len({r['id'] for r in combined})==150
    assert len({re.sub(r'\W+',' ',r['question'].casefold()).strip() for r in combined})==150
    write(base/'qa_candidates.jsonl',combined)
    report=dict(status='DRAFT_NOT_FROZEN',total=len(combined),legacy_single=len(old),new_single=len(new),approved_bridge=len(bridge),
                new_evidence_spans=sum(len(q['evidence_spans']) for q in new),
                new_answer_number_flags={q['id']:q['numeric_review_hints'] for q in new if q['numeric_review_hints']},
                legacy_answer_number_flags={q['id']:q['numeric_review_hints'] for q in old if q['numeric_review_hints']},
                legacy_revision_ids=sorted(revisions),
                applied_replacement_ids=[r['id'] for r in replacements],
                numeric_check='Literal number mismatch is a review hint only. Written-out numbers, method names, ratios and rounding require reading; absence of a flag does not prove support.',
                qa_candidates_sha256=hashlib.sha256((base/'qa_candidates.jsonl').read_bytes()).hexdigest(),
                interpretation='Source spans checked for newly authored single questions. Exactly three prepared replacements were accepted by Hill; other source reviews are not independent human approval. Legacy QA remain historically exposed. Approved bridge v1 unchanged.')
    (base/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
    manifest_path=base/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['current_composition']={k:report[k] for k in ['total','legacy_single','new_single','approved_bridge']}
    manifest['applied_replacement_ids']=report['applied_replacement_ids']
    if replacements:manifest['accepted_replacements_sha256']=hashlib.sha256(replacements_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
