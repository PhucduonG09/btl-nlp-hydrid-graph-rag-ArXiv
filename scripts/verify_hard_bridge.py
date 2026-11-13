"""Kiểm tra độc lập từng câu trong bridge_hard_drafts.xlsx: A có THẬT SỰ dùng/dựa trên thứ B đề xuất không?
LLM phải trích một câu nguyên văn từ abstract A làm bằng chứng; script kiểm tra câu đó có thật trong abstract A (không tin LLM).
  python scripts/verify_hard_bridge.py                       # chỉ các dòng đã qua bộ lọc tự động (mặc định)
  python scripts/verify_hard_bridge.py --all-valid           # mọi dòng LLM valid
Đầu ra: data/eval/bridge_hard_verified.xlsx (cùng bố cục, thêm cột 17-21; dòng qua kiểm tra xếp trên). hợp_lệ để trống cho NGƯỜI duyệt."""
import sys, re, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common.llm import chat, run_parallel, cost_summary
from src.kg.llm_extract import parse_json

VERIFY = """You are a strict reviewer for a multi-hop QA benchmark. You get paper A's abstract, paper B's abstract (the target), a question and its reference answer.
Judge ONLY from the abstracts:
1. "link": does A's abstract EXPLICITLY say that A uses / builds on / adopts / extends / compares against the specific method, resource or concept that B proposes? A merely shared topic, generic phrase or similar wording is NOT a link. If link is true give "quote": ONE sentence copied verbatim from A's abstract showing it; otherwise quote = null.
2. "presupposition_ok": does the question's wording match that real relationship (it must not claim A relies on B if A does not, and must not reverse the direction)?
3. "answer_supported": is the reference answer fully supported by B's abstract (plus A only if needed), with no invented facts?
4. "needs_A": would someone who has NOT read A be unable to tell which paper the question is about, i.e. does answering require finding A first? true if A is needed.
Return JSON only: {"link": bool, "quote": str|null, "presupposition_ok": bool, "answer_supported": bool, "needs_A": bool, "reason": "<short>"}"""
def _month(pid):
    import re as _re
    m = _re.match(r"(\d{2})(\d{2})\.", pid or ""); return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def chrono_ok(idA, idB):
    """B (paper được dựa vào) không được ra đời SAU A: so tháng nộp theo arXiv ID. Thiếu ID thì không loại."""
    a, b = _month(idA), _month(idB); return a is None or b is None or b <= a


norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[“”\"']", "", s or "")).strip().lower()


def split(cell):
    m = re.match(r"\[(.*?)\]\s*(.*?)\n\n(.*)", cell or "", re.S)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, "", "")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", default=str(ROOT / "data/eval/bridge_hard_drafts.xlsx"))
    ap.add_argument("--out", default=str(ROOT / "data/eval/bridge_hard_verified.xlsx")); ap.add_argument("--all-valid", action="store_true")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--model", default=None); a = ap.parse_args()
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb0 = load_workbook(a.src); ws0 = wb0["Viết tay"]; rows = [list(r) for r in ws0.iter_rows(min_row=2, values_only=True)]
    heads = [c.value for c in ws0[1]]
    def passed(r):   # bộ lọc tự động của bước nháp
        return r[10] == "Y" and r[11] == "Y" and not r[12] and r[13] == "không" and (r[14] or 9999) > 10 and (r[15] or 9999) <= 10
    cand = [r for r in rows if (r[10] == "Y" and r[5]) and (a.all_valid or passed(r))]
    def work(r):
        _, _, absA = split(r[2]); _, _, absB = split(r[3])
        raw, u = chat([{"role": "system", "content": VERIFY}, {"role": "user", "content":
                       f"[Paper A abstract]\n{absA}\n\n[Paper B abstract]\n{absB}\n\n[Question]\n{r[5]}\n\n[Reference answer]\n{r[6]}"}], model=a.model)
        d = parse_json(raw) or {}
        q = d.get("quote"); quote_ok = bool(q) and norm(q) in norm(absA)       # chống bịa trích dẫn
        chrono = chrono_ok(split(r[2])[0], split(r[3])[0])
        ok = chrono and bool(d.get("link")) and quote_ok and bool(d.get("presupposition_ok")) and bool(d.get("answer_supported")) and bool(d.get("needs_A"))
        return dict(row=r, ok=ok, quote_ok=quote_ok, chrono=chrono, d=d, u=u)
    res, errs = run_parallel(work, cand, workers=a.workers, desc="verify"); res = [x for x in res if x]
    if not res: sys.exit(f"Tất cả {len(errs)} dòng lỗi: {errs[:2]}")
    res.sort(key=lambda x: (not x["ok"], not x["d"].get("link")))
    wb = Workbook = None
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active; ws.title = "Viết tay"
    H = heads + ["verify OK", "link thật", "trích dẫn A (đã kiểm tra nguyên văn)", "tiền đề đúng / đáp án có căn cứ / cần A", "lý do reviewer"]
    W = [5, 18, 55, 55, 11, 55, 55, 28, 55, 9, 9, 8, 10, 12, 10, 8, 9, 9, 50, 22, 40]
    for c, (h, w) in enumerate(zip(H, W), 1):
        cell = ws.cell(1, c, h); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="305496"); ws.column_dimensions[cell.column_letter].width = w
    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True); ws.add_data_validation(dv)
    for i, x in enumerate(res, 1):
        d = x["d"]; vals = x["row"][:16] + ["OK" if x["ok"] else "", "Y" if d.get("link") and x["quote_ok"] else "N",
              (d.get("quote") if x["quote_ok"] else f"[KHÔNG NGUYÊN VĂN] {d.get('quote')}") or "",
              f"{'Y' if d.get('presupposition_ok') else 'N'} / {'Y' if d.get('answer_supported') else 'N'} / {'Y' if d.get('needs_A') else 'N'}", d.get("reason", "")]
        vals[0] = i; vals[4] = None
        for c, v in enumerate(vals, 1): ws.cell(i + 1, c, v).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[i + 1].height = 280; dv.add(f"E{i+1}")
    ws.freeze_panes = "C2"; wb.save(a.out)
    n_ok = sum(x["ok"] for x in res)
    print(json.dumps({"đã kiểm tra": len(res), "B ra đời sau A (loại)": sum(not x["chrono"] for x in res), "link thật + trích dẫn nguyên văn": sum(x["quote_ok"] and bool(x["d"].get("link")) for x in res), "qua toàn bộ kiểm tra": n_ok, "lỗi": len(errs)}, ensure_ascii=False), "|", cost_summary([x["u"] for x in res]))
    print("->", a.out)


if __name__ == "__main__":
    main()
