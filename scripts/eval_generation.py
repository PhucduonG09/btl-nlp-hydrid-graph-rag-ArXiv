"""Sinh câu trả lời có trích dẫn từ top-k của từng hệ thống rồi chấm bằng LLM-judge (model khác họ).
  python scripts/eval_generation.py --n 50 --k 5
Cần chạy run_baselines.py trước (dùng lại đúng danh sách top-k đã đo)."""
import sys, re, json, random, argparse, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
from src.common.llm import chat, run_parallel, cost_summary
from src.kg.llm_extract import parse_json

GEN = """Answer the question using ONLY the numbered sources.
Cite every claim with [n]. If the sources are insufficient, reply exactly: Insufficient information.

Sources:
{sources}

Question: {question}
Answer:"""
JUDGE = """You are a strict evaluator of a retrieval-augmented answer.
Sources:
{sources}

Question: {question}
Reference answer (may be imperfect): {reference}
Candidate answer: {answer}

1. Split the candidate answer into atomic factual claims. For each claim, decide if it is SUPPORTED by the sources above (true/false).
2. Rate relevance of the answer to the question from 1 (irrelevant) to 5 (fully on point).
3. Compare with the reference: "correct", "partial" or "incorrect".
Return JSON: {{"claims": [{{"text": "...", "supported": true}}], "relevance": 1-5, "vs_reference": "correct|partial|incorrect"}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--systems", default="bm25,dense,rrf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--judge", default=os.environ.get("LLM_JUDGE_MODEL"))
    a = ap.parse_args()
    if not a.judge:
        sys.exit("Chưa đặt LLM_JUDGE_MODEL trong .env")
    gen_model = os.environ.get("LLM_MODEL")
    papers = {}
    for l in open(ROOT / "data/processed/pilot_papers.jsonl", encoding="utf-8"):
        if l.strip():
            d = json.loads(l); papers[d["paper_id"]] = d
    qa = [json.loads(l) for l in open(ROOT / "data/eval/qa.jsonl", encoding="utf-8") if l.strip()]
    rng = random.Random(a.seed)
    sub = []
    for t in ("single", "multi"):                    # lấy mẫu phân tầng theo loại, cùng tỉ lệ
        pool = [q for q in qa if q["type"] == t]
        sub += rng.sample(pool, min(len(pool), round(a.n * len(pool) / len(qa))))
    systems = a.systems.split(",")
    ranked = {s: {} for s in systems}
    for s in systems:
        for l in open(ROOT / "results" / "runs" / f"retrieval_{s}.jsonl", encoding="utf-8"):
            d = json.loads(l); ranked[s][d["id"]] = d["ranked"]

    jobs = [(s, q) for q in sub for s in systems]

    def work(job):
        s, q = job
        ids = ranked[s][q["id"]][:a.k]
        src = "\n\n".join(f"[{n}] {papers[i]['title']} (arXiv:{i})\n{papers[i]['abstract']}"
                          for n, i in enumerate(ids, 1))
        ans, u1 = chat([{"role": "user", "content": GEN.format(sources=src, question=q["question"])}],
                       model=gen_model, max_tokens=500)
        rec = {"system": s, "id": q["id"], "type": q["type"], "answer": ans, "usage_gen": u1,
               "refused": ans.strip().startswith("Insufficient information"),
               "cites_ok": bool(re.findall(r"\[(\d+)\]", ans)) and all(1 <= int(n) <= a.k for n in re.findall(r"\[(\d+)\]", ans)),
               "gold_in_topk": bool(set(q["gold_ids"]) & set(ids))}
        if not rec["refused"]:
            jt, u2 = chat([{"role": "user", "content": JUDGE.format(sources=src, question=q["question"],
                                                                  reference=q["reference_answer"], answer=ans)}],
                          model=a.judge, max_tokens=1200)
            j = parse_json(jt) or {}
            claims = [c for c in j.get("claims", []) if isinstance(c, dict)]
            rec.update(usage_judge=u2, n_claims=len(claims),
                       faithfulness=(sum(bool(c.get("supported")) for c in claims) / len(claims)) if claims else None,
                       relevance=j.get("relevance"), vs_reference=j.get("vs_reference"))
        return rec
    res, errors = run_parallel(work, jobs, workers=a.workers, desc="generate+judge")
    if errors:
        print(f"{len(errors)} lỗi, ví dụ: {errors[0]}")
    res = [r for r in res if r]
    out = ROOT / "results" / "generation_eval.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def mean(x):
        x = [v for v in x if isinstance(v, (int, float))]
        return round(st.mean(x), 3) if x else None
    print(f"\n{len(sub)} câu x {len(systems)} hệ thống, k={a.k}, generator={gen_model}, judge={a.judge}")
    print(f"{'system':<7}{'type':<8}{'n':>4}{'refuse%':>9}{'faithful':>10}{'relev(1-5)':>12}{'correct%':>10}{'gold@k%':>9}{'cite_ok%':>9}")
    for s in systems:
        for t in ("all", "single", "multi"):
            r = [x for x in res if x["system"] == s and (t == "all" or x["type"] == t)]
            if not r:
                continue
            ok = [x for x in r if not x["refused"]]
            print(f"{s:<7}{t:<8}{len(r):>4}{100*sum(x['refused'] for x in r)/len(r):>9.0f}"
                  f"{str(mean([x.get('faithfulness') for x in ok])):>10}{str(mean([x.get('relevance') for x in ok])):>12}"
                  f"{100*sum(x.get('vs_reference')=='correct' for x in ok)/max(len(ok),1):>10.0f}"
                  f"{100*sum(x['gold_in_topk'] for x in r)/len(r):>9.0f}{100*sum(x['cites_ok'] for x in ok)/max(len(ok),1):>9.0f}")
    us = [r["usage_gen"] for r in res] + [r["usage_judge"] for r in res if "usage_judge" in r]
    print("Chi phí lượt này:", cost_summary(us))
    print("Lưu ý: n nhỏ, judge là LLM (có thiên vị), reference là bản nháp LLM; chỉ dùng để so tương đối giữa các hệ thống.")


if __name__ == "__main__":
    main()
