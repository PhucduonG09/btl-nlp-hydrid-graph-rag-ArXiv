"""LLM nháp câu bridge từ các cặp ứng viên CHƯA nằm trong bridge_handwritten.xlsx, rồi lọc tự động.
  python scripts/draft_bridge_questions.py --limit 20        # chạy thử
  python scripts/draft_bridge_questions.py                   # toàn bộ ứng viên còn lại
Đầu ra: data/eval/bridge_drafts.xlsx (cùng bố cục sheet 'Viết tay' để dùng bridge_xlsx_to_qa.py).
Bạn chỉ cần xem, sửa câu/đáp án nếu cần và đặt hợp_lệ=Y. Cột I-L là kết quả bộ lọc tự động.
Bộ lọc: (1) LLM tự phán cặp hợp lệ / homonym; (2) câu hỏi không chứa bridge_term; (3) A-only: LLM chỉ có abstract A
không được trả lời được; (4) BM25 triviality: B có vào top-10 chỉ từ câu hỏi không."""
import sys, re, json, csv, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common.llm import chat, run_parallel, cost_summary
from src.kg.llm_extract import parse_json
from scripts.bridge_triviality import load_bm25, ranks

GEN = """You help build a multi-hop QA benchmark over arXiv abstracts.
Given paper A, paper B and a candidate bridge term T: B introduces T (a dataset, benchmark, method or model), A uses T.
Step 1 - decide "valid": true only if T in A refers to the SAME thing that B introduces (not a homonym/acronym clash, and A really uses it rather than just comparing against it). Otherwise false.
Step 2 - if valid, write ONE natural question and a reference answer:
- The question describes paper A well enough to identify it (its problem/method) WITHOUT naming T, its expansion, or any alias of T.
- It asks something that only paper B's abstract answers (how B was built, why, what it shows, key numbers).
- Refer to B ONLY generically ("a benchmark it evaluates on", "the method it builds on"). Do NOT add distinguishing words of B's content (domain, task format, size, year): they make B retrievable by keyword search without ever reading A, which defeats the purpose.
- The answer uses only facts in the two abstracts. 1-3 sentences.
Return JSON only: {"valid": true/false, "reason": "<short>", "question": "...", "answer": "..."} (question/answer empty if not valid)."""

ONLY_A = """Below is an abstract and a question. Using ONLY the abstract (no outside knowledge), can the question be answered substantively?
Return JSON only: {"answerable": true/false}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=None); ap.add_argument("--cands", default=str(ROOT / "results/analysis/bridge_candidates.csv")); ap.add_argument("--pool", nargs="+", default=[str(ROOT / "data/processed/slice_c1_3000.jsonl")])
    ap.add_argument("--no-skip-done", action="store_true"); ap.add_argument("--out", default=str(ROOT / "data/eval/bridge_drafts.xlsx")); a = ap.parse_args()
    corpus = ROOT / "data/processed/slice_c1_3000.jsonl"
    P = {}
    for fn in a.pool:
        for l in open(fn, encoding="utf-8"):
            if l.strip(): d = json.loads(l); P.setdefault(d["paper_id"], d)
    rows = list(csv.DictReader(open(a.cands, encoding="utf-8-sig")))
    hand = list(csv.DictReader(open(ROOT / "results/analysis/bridge_candidates.csv", encoding="utf-8-sig")))
    done = set() if a.no_skip_done else {(r["A_id"], r["B_id"]) for r in hand if r["B_introduces"] == "1"}      # đã nằm trong sheet viết tay
    rows = [r for r in rows if (r["A_id"], r["B_id"]) not in done]
    if a.limit: rows = rows[:a.limit]
    ids, bm25 = load_bm25(a.pool)
    fmt = lambda p: f"Title: {p['title']}\nAbstract: {p['abstract']}"

    def work(r):
        A, B, t = P[r["A_id"]], P[r["B_id"]], r["bridge_term"]
        raw, u1 = chat([{"role": "system", "content": GEN},
                        {"role": "user", "content": f"T = {t}\n\n[Paper A]\n{fmt(A)}\n\n[Paper B]\n{fmt(B)}"}], model=a.model)
        d = parse_json(raw) or {}; us = [u1]
        out = {**r, "valid": bool(d.get("valid")), "reason": d.get("reason", ""), "q": (d.get("question") or "").strip(), "ans": (d.get("answer") or "").strip(),
               "has_term": "", "a_only": "", "B_rank": "", "A_rank": ""}
        if out["valid"] and out["q"]:
            out["has_term"] = bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", out["q"], re.I))
            raw2, u2 = chat([{"role": "system", "content": ONLY_A},
                             {"role": "user", "content": f"Abstract:\n{A['abstract']}\n\nQuestion: {out['q']}"}], model=a.model); us.append(u2)
            out["a_only"] = bool((parse_json(raw2) or {}).get("answerable"))
            out["B_rank"], out["A_rank"] = ranks(ids, bm25, out["q"], [r["B_id"], r["A_id"]])
        out["_u"] = us; return out

    res, errs = run_parallel(work, rows, workers=a.workers, desc="draft")
    res = [x for x in res if x]
    usages = [u for x in res for u in x.pop("_u")]
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook(); ws = wb.active; ws.title = "Viết tay"
    heads = ["#", "bridge_term (t)", "Paper A (dùng t)", "Paper B (giới thiệu t)", "hợp_lệ", "question", "reference_answer", "ghi chú",
             "LLM valid", "câu chứa term?", "A-only trả lời được?", "B_rank (BM25)", "A_rank (BM25)"]
    widths = [5, 16, 60, 60, 11, 55, 55, 30, 10, 11, 12, 11, 11]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="305496")
        ws.column_dimensions[cell.column_letter].width = w
    dv = DataValidation(type="list", formula1='"Y,N,HOMONYM"', allow_blank=True); ws.add_data_validation(dv)
    # thứ tự: hợp lệ + qua bộ lọc trước, để bạn duyệt những câu tốt nhất trước
    def good(x): return x["valid"] and x["q"] and not x["has_term"] and not x["a_only"]
    res.sort(key=lambda x: (not good(x), (x["B_rank"] or 9999) <= 10, (x["A_rank"] or 9999) > 10, not x["valid"]))   # tốt nhất: B khó tìm thẳng, A nhận ra được
    stats = {"tổng": len(res), "LLM valid": sum(x["valid"] for x in res), "qua lọc 2+3": sum(good(x) for x in res),
             "qua lọc + B_rank>10": sum(good(x) and (x["B_rank"] or 9999) > 10 for x in res)}
    for i, x in enumerate(res, 1):
        A, B = P[x["A_id"]], P[x["B_id"]]
        vals = [i, x["bridge_term"], f"[{A['paper_id']}] {A['title']}\n\n{A['abstract']}", f"[{B['paper_id']}] {B['title']}\n\n{B['abstract']}",
                "", x["q"], x["ans"], x["reason"], "Y" if x["valid"] else "N", {True: "CÓ", False: "", "": ""}[x["has_term"]],
                {True: "CÓ (loại)", False: "không", "": ""}[x["a_only"]], x["B_rank"], x["A_rank"]]
        for c, v in enumerate(vals, 1): ws.cell(i + 1, c, v).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[i + 1].height = 300 if x["valid"] else 60; dv.add(f"E{i+1}")
    ws.freeze_panes = "C2"; wb.save(a.out)
    print(json.dumps(stats, ensure_ascii=False), "| lỗi:", len(errs), "|", cost_summary(usages)); print("->", a.out)


if __name__ == "__main__":
    main()
