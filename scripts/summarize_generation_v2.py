"""Offline DEV report, all-attempt token ledger, and pending human calibration."""
import csv,hashlib,json,sys
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.answer_metrics import latest_records,summarize_answers,paired_correctness_ci
from src.eval.human_calibration import ratings,resolved_ratings_v2
from src.eval.run_status import summarize_records
from src.eval.calibration_metrics import agreement_metrics,judge_support_label
BASE=ROOT/'results/benchmark_v2_assisted/dev_01/generation_prepared'

def load(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    run=BASE/'claude5_controlled';qa=load(ROOT/'data/eval/benchmark_v2_assisted/qa_dev.jsonl')
    gen_rows=load(run/'generation.jsonl');judge_rows=load(run/'judge.jsonl')
    gen=latest_records(gen_rows);judge=latest_records(judge_rows)
    keys={(q['id'],s) for q in qa for s in ['rrf','graph']}
    assert set(gen)==set(judge)==keys and len(keys)==244
    assert all(not r.get('error') and r.get('answer') for r in gen.values())
    assert all(not r.get('error') for r in judge.values())
    assert all(not r['model_label_mismatch'] for r in list(gen.values())+list(judge.values()))
    groups={}
    for kind in ['all','single','bridge','legacy_single','new_authored_draft']:
        selected=[q for q in qa if kind=='all' or q['type']==kind or q.get('origin')==kind]
        if not selected:continue
        groups[kind]={s:summarize_answers([gen[(q['id'],s)] for q in selected],[judge[(q['id'],s)] for q in selected]) for s in ['rrf','graph']}
    diagnostic=latest_records(load(BASE/'oracle_A_only/judge.jsonl'))
    assert len(diagnostic)==31 and all(not r.get('error') for r in diagnostic.values())
    flagged=[r['id'] for r in diagnostic.values() if r['quality']=='correct' and r.get('faithfulness_raw')==1 and not r['supported_claims_with_invalid_quote']]
    a_only=dict(n=31,quality_counts=dict(Counter(r['quality'] for r in diagnostic.values())),
        candidates_for_A_only_sufficiency_review=flagged,
        interpretation='Same prompt/model with gold A only. Automatic judge, not human validation or primary retrieval. Do not modify frozen QA from this diagnostic.')
    report=dict(dataset='benchmark_v2_assisted',split='DEV',n_questions=122,n_answers=244,
        generation_sha256=sha(run/'generation.jsonl'),judge_sha256=sha(run/'judge.jsonl'),
        automatic_uncalibrated=True,test_evaluated=False,groups=groups,
        exploratory_paired_correctness={kind:paired_correctness_ci([q for q in qa if kind=='all' or q['type']==kind],judge) for kind in ['all','single','bridge']},
        a_only_diagnostic=a_only,
        limits=['DEV selected retrieval configurations; not confirmatory held-out results.',
            'Benchmark source curation includes assistant review; not all independent human labels.',
            'Faithfulness is a judge claim-support score on nonabstaining answers. Literal quote checks are a separate witness diagnostic, not proof of entailment.',
            'Provider model labels are recorded; labels alone cannot prove actual model weights.',
            'Bridge questions were screened for BM25 difficulty; conclusions are scoped to this sample.'])
    dump(run/'evaluation_summary.json',report)
    dump(run/'judge_witness_review.json',[dict(id=r['id'],system=r['system'],quality_auto=r['quality'],
        claims=[c for c in r['claims'] if c['supported'] and not c['evidence_quote_valid']])
        for r in judge.values() if r['supported_claims_with_invalid_quote']])
    # Read each public run log once. Include invalid attempts, exclude cache hits.
    ledger={};totals=defaultdict(lambda:dict(input_tokens=0,output_tokens=0,calls=0))
    paths=[BASE/'generation.jsonl',BASE/'judge.jsonl',run/'generation.jsonl',run/'judge.jsonl',
        BASE/'oracle_A_only/generation.jsonl',BASE/'oracle_A_only/judge.jsonl',ROOT/'results/demo_v2/queries.jsonl',ROOT/'results/analysis/generator_label_probe.json']
    for path in paths:
        rows=json.loads(path.read_text(encoding='utf-8')) if path.suffix=='.json' else load(path)
        stats=summarize_records(rows) if all('id' in r and 'system' in r for r in rows) else None
        billed=[r['usage'] for r in rows if r.get('usage') and not r['usage'].get('cached')]
        for u in billed:
            model=u.get('served_model') or 'unknown';bucket=totals[model]
            bucket['input_tokens']+=u.get('in',0);bucket['output_tokens']+=u.get('out',0);bucket['calls']+=1
        ledger[str(path.relative_to(ROOT))]=dict(sha256=sha(path),records=len(rows),run_status=stats,
            provider_input_tokens=sum(u.get('in',0) for u in billed),provider_output_tokens=sum(u.get('out',0) for u in billed))
    dump(run/'token_ledger.json',dict(scope='This generation/judging batch, initial label-drift pilot, label probes, A-only diagnostic and one demo call. Excludes historical extraction and embedding.',
        pricing_status='Not supplied; do not infer USD or VND. Provider-reported usage is not an invoice and may omit failed requests without a usage response.',
        usd=None,vnd=None,by_served_model=dict(totals),files=ledger))
    pack=ROOT/'data/eval/generation_calibration_v2';items=load(pack/'blind_items.jsonl')
    manifest=json.loads((pack/'manifest.json').read_text(encoding='utf-8'));assert sha(pack/'blind_items.jsonl')==manifest['items_sha256']
    assert sha(run/'generation.jsonl')==manifest['generation_file_sha256']
    human,pending=resolved_ratings_v2(ratings(pack/'ratings.sqlite3'),{r['sample_id'] for r in items})
    key={r['sample_id']:r for r in json.loads((run/'human_calibration_key.json').read_text(encoding='utf-8'))}
    confusion=Counter()
    for sample,h in human.items():
        k=key[sample];confusion[(h['quality'],judge[(k['question_id'],k['system'])]['quality'])]+=1
    n=sum(confusion.values());observed=sum(v for (a,b),v in confusion.items() if a==b)/n if n else None
    labels=['correct','partial','incorrect','abstained']
    expected=sum(sum(v for (a,b),v in confusion.items() if a==c)*sum(v for (a,b),v in confusion.items() if b==c) for c in labels)/(n*n) if n else None
    macro_f1=None
    if n:
        f1=[]
        for c in labels:
            denom=sum(v for (a,b),v in confusion.items() if a==c)+sum(v for (a,b),v in confusion.items() if b==c)
            f1.append(2*confusion[(c,c)]/denom if denom else 0)
        macro_f1=sum(f1)/len(f1)
    calibration=dict(n_items=50,n_resolved=n,pending_counts=dict(Counter(pending.values())),pending=pending,
        complete=n==50,agreement=observed,cohen_kappa=(observed-expected)/(1-expected) if n and expected<1 else None,
        macro_f1_fixed_four_classes=macro_f1,
        confusion=[dict(human=a,judge=b,n=v) for (a,b),v in sorted(confusion.items())],
        policy='Explicit named human ratings only. Any disagreement or uncertain label remains pending; reference_issue is a stratum flag (amendment A1-2026-11-11). A single named rater permits calibration but does not establish independent double annotation.')
    calibration['quality_comparison']=agreement_metrics([(h['quality'],judge[(key[sample]['question_id'],key[sample]['system'])]['quality']) for sample,h in human.items()],labels)
    calibration['support_comparison']=agreement_metrics([(h['support'],judge_support_label(judge[(key[sample]['question_id'],key[sample]['system'])])) for sample,h in human.items()],['all_supported','mixed','unsupported','not_applicable'])
    calibration['support_interpretation']='Source-support category agreement, not a correlation with per-claim human scores. Quote validity remains a separate diagnostic.'
    strata={}
    for name,flag in (('reference_supported',False),('reference_unsupported',True)):
        sub=[(h['quality'],judge[(key[sample]['question_id'],key[sample]['system'])]['quality']) for sample,h in human.items() if h['reference_issue']==flag]
        strata[name]=dict(n=len(sub),quality_comparison=agreement_metrics(sub,labels) if sub else None)
    calibration['reference_issue_strata']=strata
    dump(run/'human_calibration_status.json',calibration)
    with (run/'answers_for_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['id','type','system','quality_auto','faithfulness_auto','refused','gold_all_in_context','invalid_citations','supported_claims_invalid_quote']);writer.writeheader()
        for q in qa:
            for s in ['rrf','graph']:
                g=gen[(q['id'],s)];j=judge[(q['id'],s)]
                writer.writerow(dict(id=q['id'],type=q['type'],system=s,quality_auto=j['quality'],faithfulness_auto=j['faithfulness_raw'],refused=g['refused'],gold_all_in_context=g['gold_all_in_context'],invalid_citations=len(g['invalid_citations']),supported_claims_invalid_quote=j['supported_claims_with_invalid_quote']))
    lines=['# Kết quả generation DEV v2 — tự động, chưa hiệu chuẩn người','',
        '122 câu DEV × 2 hệ thống. Generator: claude-sonnet-5; judge: gpt-6-sol. Tập test 28 câu chưa đánh giá.',
        '', '| Nhóm | Hệ thống | Đúng / tổng | Đúng một phần | Từ chối (judge) | Faithfulness, bỏ đáp án từ chối |',
        '|---|---|---:|---:|---:|---:|']
    for kind,systems in groups.items():
        for s,r in systems.items():
            value=r['faithfulness_macro_nonabstaining'];value=f'{value:.3f}' if value is not None else 'N/A'
            lines.append(f"| {kind} | {s} | {r['quality_counts'].get('correct',0)}/{r['n']} | {r['quality_counts'].get('partial',0)} | {r['quality_counts'].get('abstained',0)} | {value} |")
    lines+=['','Không dùng độ hợp lệ cú pháp trích dẫn làm bằng chứng câu trả lời đúng. Các chỉ số trên do AI chấm, chưa phải kết quả đã xác nhận bởi người.',
        '',f"Chỉ cấp A: {a_only['quality_counts']}. Các trường hợp nghi A-only đủ cần rà: {flagged}.",
        '',f'Gói chấm người: {n}/50 mẫu đã chốt. Mở http://127.0.0.1:8503 để chấm; không gọi API.',
        '', 'Token theo từng nhãn model và mọi lượt thử được lưu trong token_ledger.json; chưa quy đổi tiền. File trả lời gốc, kết quả judge, lịch sử lỗi đều được giữ.',
        '', 'Hạn chế: cấu hình chọn trên DEV; câu bridge có thiên lệch lọc BM25; MENTIONS chưa phải quan hệ ngữ nghĩa USES/PROPOSES; nguồn nhãn assisted không thay thế annotation người.']
    (run/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(groups=groups,a_only=a_only,human_resolved=n,token_totals=dict(totals)),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
