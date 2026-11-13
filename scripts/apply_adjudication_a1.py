"""Áp dụng nhãn đã chốt sau đối chiếu nguồn (Amendment A1, 2026-11-11) cho các dòng của Phúc.

Chạy trên máy của bạn (Windows), KHÔNG chạy trong lúc giao diện chấm đang mở:
  .venv-kg/Scripts/python.exe -X utf8 scripts/apply_adjudication_a1.py
Không gọi API, không mở qa_test. Chỉ sửa dòng reviewer 'Phúc'; dòng duyanh giữ nguyên.
Nhãn gốc nằm trong ratings.before-adjudication.sqlite3 (dùng tính kappa trước thảo luận).
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.human_calibration import save_rating,ratings,resolved_ratings_v2
PACK=ROOT/'data/eval/generation_calibration_v2';DB=PACK/'ratings.sqlite3'
TAG='[Điều chỉnh sau đối chiếu nguồn 2026-11-11; nhập theo xác nhận của người dùng. Nhãn gốc: {old}] '
EDITS=[('H022','correct','mixed',0),('H032','correct','mixed',0),('H004','abstained','not_applicable',1),
       ('H042','correct','all_supported',0),('H050','correct','mixed',0),('H017','correct','all_supported',0)]
def main():
    if not (PACK/'ratings.before-adjudication.sqlite3').exists():raise SystemExit('Thiếu bản sao lưu ratings.before-adjudication.sqlite3, dừng.')
    rows={(r['sample_id'],r['reviewer']):r for r in ratings(DB)};log=[]
    for s,q,su,ri in EDITS:
        o=rows[(s,'Phúc')]
        if (o['quality'],o['support'],int(o['reference_issue']))==(q,su,ri):print('đã đúng, bỏ qua',s);continue
        old=f"{o['quality']}/{o['support']}/ref={o['reference_issue']}"
        save_rating(DB,sample_id=s,reviewer='Phúc',quality=q,support=su,note=TAG.format(old=old)+(o['note'] or ''),reference_issue=bool(ri))
        log.append(dict(sample_id=s,reviewer='Phúc',before=old,after=f'{q}/{su}/ref={ri}'))
    (PACK/'adjudication_log.json').write_text(json.dumps(dict(date='2026-11-11',backup='ratings.before-adjudication.sqlite3',changes=log,
        note='Chỉ sửa dòng Phúc; duyanh không đổi. Cờ reference_issue của H006 giữ nguyên.'),ensure_ascii=False,indent=2),encoding='utf-8')
    ids={json.loads(l)['sample_id'] for l in (PACK/'blind_items.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()}
    r,p=resolved_ratings_v2(ratings(DB),ids)
    print(f'Đã sửa {len(log)} dòng. Resolved {len(r)}/50 (cờ reference_issue: {sum(v["reference_issue"] for v in r.values())}). Pending: {p or "không"}')
main()
