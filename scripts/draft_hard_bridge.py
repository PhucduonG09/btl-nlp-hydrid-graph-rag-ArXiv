"""LLM nháp câu bridge KHÓ (chain2 và concept) từ results/analysis/hard_bridge_candidates.csv, rồi lọc tự động.
  python scripts/draft_hard_bridge.py --limit 30                      # thử
  python scripts/draft_hard_bridge.py --kind concept --limit 300     # đợt lớn (≈2,9k token/cặp)
Đầu ra: data/eval/bridge_hard_drafts.xlsx (sheet 'Viết tay', cột 1-8 như bridge_handwritten.xlsx; cột 9 = Paper C, cột 10 = kind,
cột 11-16 = kết quả bộ lọc). Bạn đặt hợp_lệ=Y rồi chạy bridge_xlsx_to_qa.py --xlsx ... (chain2 → gold 3 paper, type 'bridge_chain2').
Lọc: (1) LLM phán hợp lệ + 'unique'; (2) câu không chứa term/term2; (3) A-only không trả lời được; (4) BM25 trên corpus v2:
paper đích (B với concept, C với chain2) KHÔNG vào top-10 chỉ từ câu hỏi, A vào top-10 (A nhận ra được)."""
import sys, re, json, csv, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common.llm import chat, run_parallel, cost_summary
from src.kg.llm_extract import parse_json
from scripts.bridge_triviality import load_bm25, ranks
from scripts.run_baselines import tok

COMMON = """- The question must read like a real user question, one or two sentences.
- Do NOT name the bridge term(s), their expansions, synonyms or distinctive parts of them.
- Refer to the target paper ONLY by its role ("the technique it relies on for X"), never by topic words that identify it (domain, task format, dataset size, year): such words make the target retrievable by keyword search without reading paper A.
- The answer uses only facts in the abstracts, 1-3 sentences."""
GEN = {
 "concept": """You help build a multi-hop QA benchmark over arXiv abstracts.
Paper A relies on a concept T that also appears in the title of paper B (T is a specific phrase, not a named artifact).
Step 1 - "valid": true only if A really uses/builds on the SAME concept T that B studies (same meaning, not coincidental wording).
Step 2 - "unique": true only if B is plausibly THE paper a reader should consult about T for A (not one of many equivalent papers).
Step 3 - if valid and unique, write ONE question about what only paper B's abstract answers (what B proposes, shows, how it measures it), phrased by describing A.
""" + COMMON + """
Return JSON only: {"valid": true/false, "unique": true/false, "reason": "<short>", "question": "...", "answer": "..."} (question/answer empty if not valid or not unique).""",
 "chain2": """You help build a multi-hop QA benchmark over arXiv abstracts.
Chain A -> B -> C: A mentions E1, which B introduces; B mentions E2, which C introduces.
Step 1 - "valid": true only if BOTH links are real (A uses what B introduces AND B uses what C introduces; neither is a homonym/acronym clash).
Step 2 - "unique": true if the chain identifies C unambiguously.
Step 3 - if valid, write ONE question that describes A and asks something only paper C's abstract answers, reachable only through A -> B -> C (e.g. "the resource that the tool it relies on itself builds on").
""" + COMMON + """
Return JSON only: {"valid": true/false, "unique": true/false, "reason": "<short>", "question": "...", "answer": "..."}."""}
REWRITE = """You rewrite a multi-hop question so that keyword search over arXiv abstracts can NOT find the target paper from the question alone.
The question must still (1) identify paper A (describe A's main problem/approach in other words) and (2) ask for what only the target paper's abstract answers.
Do not use the listed words or their close synonyms, and do not describe the target paper's topic; refer to it only by its role ("the approach it builds on / compares against / relies on").
Return JSON only: {"question": "...", "answer": "..."} (answer from the abstracts; keep it consistent with the new question)."""
ONLY_A = """Below is an abstract and a question. Using ONLY the abstract (no outside knowledge), can the question be answered substantively?
Return JSON only: {"answerable": true/false}"""


def has(q, t):
    return bool(t) and bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![A-Za-z0-9])", q, re.I))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--kind", choices=["concept", "chain2", "all"], default="concept")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--rewrites", type=int, default=2, help="số vòng viết lại khi đích còn tìm thẳng được từ câu hỏi"); ap.add_argument("--model", default=None)
    ap.add_argument("--cands", default=str(ROOT / "results/analysis/hard_bridge_candidates.csv"))
    ap.add_argument("--corpus", default=str(ROOT / "data/processed/benchmark_v2_corpus.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data/eval/bridge_hard_drafts.xlsx")); a = ap.parse_args()
    P = {}
    for l in open(a.corpus, encoding="utf-8"):
        if l.strip(): d = json.loads(l); P[d["paper_id"]] = d
    rows = list(csv.DictReader(open(a.cands, encoding="utf-8-sig")))
    if a.kind != "all": rows = [r for r in rows if r["kind"] == a.kind]
    if a.limit:
        if a.kind == "all":      # chia đều hai loại
            rows = [r for k in ("chain2", "concept") for r in [x for x in rows if x["kind"] == k][: a.limit // 2]]
        else: rows = rows[: a.limit]
    ids, bm25 = load_bm25(a.corpus)
    fmt = lambda p: f"Title: {p['title']}\nAbstract: {p['abstract']}"

    import traceback

    def work(r):
        try:
            return _work(r)
        except Exception:                                   # in traceback đầy đủ thay vì nuốt lỗi
            if not getattr(work, "shown", False):
                work.shown = True; print("\n[LỖI đầu tiên]\n" + traceback.format_exc(), file=sys.stderr)
            raise

    def _work(r):
        A, B = P[r["A_id"]], P[r["B_id"]]; C = P.get(r["C_id"]) if r["C_id"] else None
        body = f"T = {r['term']}\n\n[Paper A]\n{fmt(A)}\n\n[Paper B]\n{fmt(B)}" if not C else \
               f"E1 = {r['term']}  (introduced by B)\nE2 = {r['term2']}  (introduced by C)\n\n[Paper A]\n{fmt(A)}\n\n[Paper B]\n{fmt(B)}\n\n[Paper C]\n{fmt(C)}"
        raw, u1 = chat([{"role": "system", "content": GEN[r["kind"]]}, {"role": "user", "content": body}], model=a.model)
        d = parse_json(raw) or {}; us = [u1]; target = r["C_id"] or r["B_id"]
        o = {**r, "valid": bool(d.get("valid")), "unique": bool(d.get("unique")), "reason": d.get("reason", ""), "q": (d.get("question") or "").strip(),
             "ans": (d.get("answer") or "").strip(), "has_term": "", "a_only": "", "T_rank": "", "A_rank": ""}
        if o["valid"] and o["unique"] and o["q"]:
            def check():
                o["has_term"] = has(o["q"], r["term"]) or has(o["q"], r["term2"])
                raw2, u2 = chat([{"role": "system", "content": ONLY_A}, {"role": "user", "content": f"Abstract:\n{A['abstract']}\n\nQuestion: {o['q']}"}], model=a.model); us.append(u2)
                o["a_only"] = bool((parse_json(raw2) or {}).get("answerable"))
                o["T_rank"], o["A_rank"] = ranks(ids, bm25, o["q"], [target, r["A_id"]])
            check(); o["rewrites"] = 0; Tp = P[target]; ttext = set(tok(Tp["title"] + " " + Tp["abstract"]))
            while a.rewrites > o["rewrites"] and (o["T_rank"] or 9999) <= 10:         # đích vẫn tìm thẳng được: viết lại có phản hồi
                leak = sorted(set(tok(o["q"])) & ttext, key=lambda w: -bm25.idf.get(w, 0))[:10]
                raw3, u3 = chat([{"role": "system", "content": REWRITE}, {"role": "user", "content":
                                 f"Paper A abstract:\n{A['abstract']}\n\nTarget paper abstract:\n{Tp['abstract']}\n\nCurrent question: {o['q']}\n"
                                 f"Target currently ranks #{o['T_rank']} from the question alone. Words shared with the target (avoid them): {', '.join(leak)}"}], model=a.model); us.append(u3)
                d3 = parse_json(raw3) or {}
                if not d3.get("question"): break
                o["q"], o["ans"] = d3["question"].strip(), (d3.get("answer") or o["ans"]).strip(); o["rewrites"] += 1; check()
        o["_u"] = us; return o

    res, errs = run_parallel(work, rows, workers=a.workers, desc="draft-hard"); res = [x for x in res if x]
    if not res:
        print(f"\nTất cả {len(errs)} cặp đều lỗi, không ghi file. Mẫu lỗi: {errs[:2]}", file=sys.stderr); sys.exit(1)
    usages = [u for x in res for u in x.pop("_u")]
    good = lambda x: bool(x["valid"] and x["unique"] and x["q"]) and not x["has_term"] and not x["a_only"]
    hard = lambda x: (x["T_rank"] or 9999) > 10 and (x["A_rank"] or 9999) <= 10
    res.sort(key=lambda x: (not good(x), not hard(x), not x["valid"]))
    stats = {"tổng": len(res), "valid": sum(x["valid"] for x in res), "valid+unique": sum(x["valid"] and x["unique"] for x in res),
             "qua lọc 2+3": sum(good(x) for x in res), "qua lọc + đích khó (T_rank>10, A_rank<=10)": sum(good(x) and hard(x) for x in res)}
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook(); ws = wb.active; ws.title = "Viết tay"
    heads = ["#", "term", "Paper A", "Paper B", "hợp_lệ", "question", "reference_answer", "ghi chú", "Paper C (chain2)", "kind",
             "LLM valid", "unique", "câu chứa term?", "A-only trả lời được?", "T_rank (đích)", "A_rank"]
    widths = [5, 18, 55, 55, 11, 55, 55, 28, 55, 9, 9, 8, 10, 12, 10, 8]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="305496"); ws.column_dimensions[cell.column_letter].width = w
    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True); ws.add_data_validation(dv)
    yn = {True: "CÓ", False: "", "": ""}
    for i, x in enumerate(res, 1):
        A, B = P[x["A_id"]], P[x["B_id"]]; C = P.get(x["C_id"]) if x["C_id"] else None
        vals = [i, x["term"] + (f" / {x['term2']}" if x["term2"] else ""), f"[{A['paper_id']}] {A['title']}\n\n{A['abstract']}", f"[{B['paper_id']}] {B['title']}\n\n{B['abstract']}", "",
                x["q"], x["ans"], x["reason"], (f"[{C['paper_id']}] {C['title']}\n\n{C['abstract']}" if C else ""), x["kind"], "Y" if x["valid"] else "N", "Y" if x["unique"] else "N",
                yn[x["has_term"]], {True: "CÓ (loại)", False: "không", "": ""}[x["a_only"]], x["T_rank"], x["A_rank"]]
        for c, v in enumerate(vals, 1): ws.cell(i + 1, c, v).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[i + 1].height = 300 if x["valid"] else 60; dv.add(f"E{i+1}")
    ws.freeze_panes = "C2"; wb.save(a.out)
    print(json.dumps(stats, ensure_ascii=False), "| lỗi:", len(errs), "|", cost_summary(usages)); print("->", a.out)


if __name__ == "__main__":
    main()
