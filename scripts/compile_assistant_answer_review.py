"""Validate 50 manually source-read assistant decisions; never writes human labels."""
import csv,hashlib,json,sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kg.evidence import locate_evidence
BASE=ROOT/'data/eval/generation_calibration_v2';OUT=BASE/'assistant_review_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pack=BASE/'blind_items.jsonl';manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    assert sha(pack)==manifest['items_sha256']
    items={r['sample_id']:r for r in map(json.loads,pack.read_text(encoding='utf-8').splitlines())}
    decisions=json.loads((OUT/'decisions.json').read_text(encoding='utf-8'))
    assert len(decisions)==50 and len({r['sample_id'] for r in decisions})==50
    assert {r['sample_id'] for r in decisions}==set(items)
    compiled=[]
    for r in decisions:
        assert r['quality'] in {'correct','partial','incorrect','abstained'}
        assert r['support'] in {'all_supported','mixed','unsupported','not_applicable'}
        assert r['priority'] in {'none','low','high'} and r['note'].strip()
        assert (r['quality']=='abstained')==(r['support']=='not_applicable')
        item=items[r['sample_id']];sources={s['number']:s for s in item['sources']};evidence=[]
        for number,quote in r['witnesses']:
            s=sources[number];text=s['title']+'\n'+s['abstract'];span=locate_evidence(text,quote)
            assert span is not None,(r['sample_id'],number,quote)
            evidence.append(dict(source_number=number,paper_id=s['paper_id'],**asdict(span)))
        compiled.append(dict(r,reviewer_type='assistant_source_review',human_label=False,
            source_item_sha256=hashlib.sha256(json.dumps(item,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
            verified_source_spans=evidence))
    result=dict(version='assistant-answer-review-v1',n_reviewed=50,
        input_pack_sha256=sha(pack),decisions_sha256=sha(OUT/'decisions.json'),
        reviewer='Assistant reading the questions, answers, reference answers and cited/relevant supplied abstracts. Decisions are authored individually, not copied from the automatic judge.',
        human_labels_written=0,original_model_answers_modified=False,frozen_qa_modified=False,
        quality_counts=dict(Counter(r['quality'] for r in compiled)),
        priority_counts=dict(Counter(r['priority'] for r in compiled)),
        hill_question_decisions_pending=[r['sample_id'] for r in compiled if r.get('question_needs_review')],
        verified_witness_count=sum(len(r['verified_source_spans']) for r in compiled),
        scope='Source-constrained assistant review of 50 DEV responses. Not independent human calibration, no API calls, no Excel changes. A valid refusal still does not answer the benchmark question. Partial/incorrect responses are preserved as experimental failures.',
        items=compiled)
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['sample_id','quality','support','priority','question_needs_review','note','proposed_revision']
    with (OUT/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(compiled)
    lines=['# Rà 50 đáp án theo nguồn — nhận xét của trợ lý','',
        'Đã đọc 50 câu hỏi, đáp án, reference và đối chiếu các nguồn liên quan trong context. Không dùng kiến thức bên ngoài để bù paper bị thiếu. Không thay câu trả lời đã sinh, gold, workbook hoặc nhãn người.',
        '',f"Kết quả rà: {result['quality_counts']}. Mức ưu tiên: {result['priority_counts']}.",
        '', '## Năm mẫu cần xem trước','', '| Mẫu | Kết luận | Vấn đề |','|---|---|---|']
    lines += [f"| {r['sample_id']} | {r['quality']} | {r['note']} |" for r in compiled if r['priority']=='high']
    lines += ['', '## Năm góp ý nhỏ','']+[f"- **{r['sample_id']}**: {r['note']}" for r in compiled if r['priority']=='low']
    lines += ['', '## Trạng thái cần phân biệt','',
        'H036 cần Hill chốt tính duy nhất của câu hỏi gốc; chưa tự sửa phiên bản benchmark đóng băng. Những mẫu khác có lỗi/thiếu hop là kết quả thực nghiệm cần giữ, không phải dòng dữ liệu cần sửa cho tăng điểm.',
        '', 'Có 13 đáp án từ chối: hợp lý với nguồn được cấp ở lượt rà này nhưng không được tính thành 13 câu trả lời đúng. Có 33 đáp án đáp ứng ý chính; trong đó một số có góp ý nhỏ về diễn đạt hoặc trích dẫn, xem từng dòng.',
        '', 'Đây là 50 lượt rà của trợ lý, không thay thế 50 nhãn người độc lập. Chưa điền SQLite human ratings hoặc mở khóa chạy test. Nếu người chấm đã xem gợi ý trợ lý, không mô tả lượt chấm sau đó là chấm mù không tiếp xúc AI.',
        '', 'Xem trên máy: http://127.0.0.1:8503/?view=assistant . Tệp review.json có hash và bằng chứng trích nguyên văn; review.csv là bảng phụ mới, không ghi workbook cũ.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
