"""Locate quote failures without changing judge records or inventing entailment labels."""
import hashlib,json,re,sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.answer_metrics import latest_records
from src.kg.evidence import locate_evidence
BASE=ROOT/'results/benchmark_v2_assisted/dev_01/generation_prepared'
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    run=BASE/'claude5_controlled';path=run/'judge.jsonl';before=sha(path)
    jobs=latest_records(load(BASE/'jobs_dev.jsonl'));judged=latest_records(load(path));rows=[]
    for key,judge in judged.items():
        sources=jobs[key]['sources'];source_map={s['number']:s for s in sources}
        for c in judge['claims']:
            if not c['supported'] or c['evidence_quote_valid']:continue
            quote=c['evidence_quote'];s=source_map.get(c['source_number'])
            other=[p['number'] for p in sources if p['number']!=c['source_number'] and quote.strip() and locate_evidence(p['title']+'\n'+p['abstract'],quote)]
            if judge['quality']=='abstained' and s is None and not quote.strip():category='abstention_absence_statement_without_positive_witness'
            elif other:category='quote_found_in_different_source'
            elif '...' in quote or '…' in quote:category='nonliteral_ellipsis_quote'
            elif s is None:category='missing_source'
            else:category='nonliteral_quote_needs_source_review'
            candidates=[]
            if s:
                text=s['title']+'\n'+s['abstract']
                sentences=re.split(r'(?<=[.!?])\s+',text)
                for sentence in sentences:
                    ratio=SequenceMatcher(None,quote.lower(),sentence.lower(),autojunk=False).ratio()
                    candidates.append((ratio,sentence))
            nearest=sorted(candidates,reverse=True)[:2]
            rows.append(dict(id=key[0],system=key[1],judge_quality=judge['quality'],claim=c['text'],
                source_number=c['source_number'],paper_id=s['paper_id'] if s else None,
                witness_quote=quote,diagnostic_category=category,quote_found_in_other_sources=other,
                nearest_source_sentences=[dict(similarity=round(score,4),verbatim_source_sentence=text) for score,text in nearest],
                entails_claim='Not assessed. String similarity and quote location do not establish entailment.',human_reviewed=False))
    assert sha(path)==before
    report=dict(judge_file_sha256=before,answers_with_flagged_witness=len({(r['id'],r['system']) for r in rows}),
        n_claims=len(rows),categories=dict(Counter(r['diagnostic_category'] for r in rows)),
        scope='Automatic quote audit on DEV only; no API, no human labels, no modification of original judge/support scores.',items=rows)
    (run/'judge_witness_diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='items'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
