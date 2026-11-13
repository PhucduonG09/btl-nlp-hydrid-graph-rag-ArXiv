"""LLM nháp câu hỏi bridge từ cặp trích dẫn TỰ NHIÊN (data/eval/citation_bridge/selected_pairs.jsonl). GỌI API: user tự chạy.
  python -X utf8 scripts/draft_citation_bridge.py --limit 10     # thử ~20 lời gọi để xem chi phí/chất lượng
  python -X utf8 scripts/draft_citation_bridge.py                # toàn bộ 320 cặp (~640 lời gọi)
Bộ lọc TỰ ĐỘNG chỉ gồm: (1) LLM phán cặp có câu hỏi hợp lệ; (2) A-only: chỉ có abstract A thì không được trả lời được;
(3) rò tên B: câu hỏi không chứa tên riêng/viết tắt trong tiêu đề B. KHÔNG lọc theo BM25/dense/graph; B_rank chỉ ghi làm biến phụ.
Đầu ra: data/eval/citation_bridge/drafts.jsonl (mọi cặp kèm phán quyết, dùng cho bước duyệt mù)."""
import sys, re, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common.llm import chat, run_parallel, cost_summary
from src.kg.llm_extract import parse_json

GEN = """You help build a multi-hop QA benchmark over arXiv abstracts. Paper A cites paper B; we do not know why.
Step 1 - decide "valid": true only if B's abstract contains a specific fact, method, dataset or result that is NOT in A's abstract, and a natural question about paper A could only be answered by reading B. Otherwise false.
Step 2 - if valid, write ONE natural question and a reference answer:
- The question describes paper A well enough to identify it (its problem/method), then asks something that only paper B's abstract answers.
- The question MUST be about paper A: identify A by its own problem/method using words from A's abstract, so a reader could find A from the question alone.
- Refer to B ONLY through its relation to A ("the method it builds on", "a related prior approach"). Do NOT name B's title, method/dataset/model name or acronyms, and do NOT describe B's own content: no technical terms, task names, domains, numbers or features that appear in B's abstract but not in A's. The only things the question may reveal about B are its role relative to A and the property asked about.
- The answer uses only facts stated in the two abstracts. 1-3 sentences.
Do not deliberate at length. Reply immediately with the JSON object only, no prose before or after; "reason" must be under 25 words.
Return JSON only: {"valid": true/false, "reason": "<short>", "question": "...", "answer": "..."} (question/answer empty if not valid)."""
ONLY_A = """Below is an abstract and a question. Using ONLY the abstract (no outside knowledge), can the question be answered substantively?
Answer immediately with the JSON object only.
Return JSON only: {"answerable": true/false}"""
fmt = lambda p: f"Title: {p['title']}\nAbstract: {p['abstract']}"


def leaks_b_name(title, question):
    """Câu hỏi chứa tên riêng/viết tắt (>=2 chữ hoa hoặc có chữ số) trong tiêu đề B."""
    names = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9+-]*", title) if sum(c.isupper() for c in w) >= 2 or any(c.isdigit() for c in w)]
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", question, re.I) for w in names)


def draft_one(pair, P, ids, bm25, model=None, max_tokens=None):
    from scripts.bridge_triviality import ranks
    A, B = P[pair["A_id"]], P[pair["B_id"]]
    raw, u1 = chat([{"role": "system", "content": GEN}, {"role": "user", "content": f"[Paper A]\n{fmt(A)}\n\n[Paper B]\n{fmt(B)}"}], model=model, max_tokens=max_tokens)
    d = parse_json(raw) or {}; us = [u1]
    out = {**pair, "valid": bool(d.get("valid")), "reason": d.get("reason", ""), "q": (d.get("question") or "").strip(), "ans": (d.get("answer") or "").strip(),
           "leaks_B_name": None, "a_only": None, "B_rank": None, "A_rank": None}
    if out["valid"] and out["q"]:
        out["leaks_B_name"] = leaks_b_name(B["title"], out["q"])
        raw2, u2 = chat([{"role": "system", "content": ONLY_A}, {"role": "user", "content": f"Abstract:\n{A['abstract']}\n\nQuestion: {out['q']}"}], model=model, max_tokens=max_tokens); us.append(u2)
        out["a_only"] = bool((parse_json(raw2) or {}).get("answerable"))
        out["B_rank"], out["A_rank"] = ranks(ids, bm25, out["q"], [pair["B_id"], pair["A_id"]])   # biến phụ, KHÔNG dùng để lọc
    out["_u"] = us; return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=str(ROOT / "data/eval/citation_bridge/drafts.jsonl")); a = ap.parse_args()
    if Path(a.out).exists() and not a.limit: sys.exit("drafts.jsonl đã tồn tại; đổi --out hoặc xóa nếu muốn chạy lại.")
    from scripts.bridge_triviality import load_bm25
    manifest = json.loads((ROOT / "data/eval/benchmark_v2_assisted/manifest.json").read_text(encoding="utf-8")); corpus = ROOT / manifest["corpus"].replace("\\", "/")
    P = {d["paper_id"]: d for d in (json.loads(l) for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip())}; ids, bm25 = load_bm25(str(corpus))
    pairs = [json.loads(l) for l in (ROOT / "data/eval/citation_bridge/selected_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit: pairs = pairs[:a.limit]; a.out = str(Path(a.out).with_name("drafts_trial.jsonl"))
    res, errs = run_parallel(lambda p: draft_one(p, P, ids, bm25, a.model, a.max_tokens), pairs, workers=a.workers, desc="draft"); res = [x for x in res if x]
    for i, e in errs[:5]: print("LỖI", pairs[i]["pair_id"], e[:300])
    usages = [u for x in res for u in x.pop("_u")]
    Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res), encoding="utf-8")
    good = lambda x: x["valid"] and x["q"] and not x["leaks_B_name"] and not x["a_only"]
    print(json.dumps({"tổng": len(res), "LLM valid": sum(x["valid"] for x in res), "qua lọc tự động": sum(bool(good(x)) for x in res), "lỗi": len(errs)}, ensure_ascii=False), "|", cost_summary(usages)); print("->", a.out)


if __name__ == "__main__": main()
