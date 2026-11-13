"""Sinh bộ câu hỏi đánh giá. LLM viết câu hỏi nháp, bạn rà tay trong qa_review.csv rồi finalize.
  python scripts/make_qa.py generate --triples data/kg/triples_v3.jsonl --n-single 60 --n-multi 40
  (mở data/eval/qa_review.csv: đặt keep=0 cho câu dở, sửa cột question nếu cần)
  python scripts/make_qa.py finalize
LƯU Ý: câu multi-hop được chọn từ KG, nên khi so Graph-RAG ở CK sẽ có thiên lệch có lợi cho đồ thị; phải nêu trong báo cáo."""
import sys, csv, json, random, argparse, re
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.llm import chat, run_parallel, cost_summary
from src.kg.llm_extract import parse_json
from src.kg.aliases import GENERIC

EVAL = ROOT / "data" / "eval"
SINGLE = """You write evaluation questions for a retrieval system over arXiv abstracts.
Given ONE paper (title + abstract), write ONE natural question that a researcher who has NOT seen this paper might ask,
whose answer is stated in the abstract.
Rules: do not copy phrases of 4+ words from the title; never say "this paper" or "the authors".
Style requested: {style}
Return JSON: {{"question": "...", "answer": "1-2 sentences using only the abstract"}}"""
STYLES = ["conceptual: describe the problem/idea in your own words, avoid the paper's own method name",
          "specific: may use the key technical terms of the paper, but still do not copy the title"]
MULTI = """You write evaluation questions for a retrieval system over arXiv abstracts.
Given TWO papers (A and B) that both involve the concept "{concept}", write ONE question that can only be answered
well by combining information from BOTH abstracts (e.g. compare how each one relates to the concept, or ask what
two different works respectively propose). Do not mention titles; never say "paper A/B".
Return JSON: {{"question": "...", "answer": "2-3 sentences using only the two abstracts"}}"""


def lcs_run(a, b):
    a, b, best = a.lower().split(), b.lower().split(), 0
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            best = max(best, k)
    return best


def fmt(p):
    return f"Title: {p['title']}\nAbstract: {p['abstract']}"


def gen(a):
    rng = random.Random(a.seed)
    papers = [json.loads(l) for l in open(a.papers, encoding="utf-8") if l.strip()]
    byid = {p["paper_id"]: p for p in papers}
    jobs = []
    for p in rng.sample(papers, a.n_single * 2):          # lấy dư, một số câu sẽ bị loại
        jobs.append(("single", [p["paper_id"]], SINGLE.format(style=rng.choice(STYLES)), fmt(p), None))
    cp = defaultdict(set)
    for l in open(a.triples, encoding="utf-8"):
        d = json.loads(l)
        if d["relation"] in ("PROPOSES", "USES") and d["concept"] not in GENERIC:
            cp[d["concept"]].add(d["paper_id"])
    concepts = [c for c, ps in cp.items() if 2 <= len(ps) <= 6]
    rng.shuffle(concepts)
    used = defaultdict(int)
    for c in concepts:
        ps = sorted(cp[c]); rng.shuffle(ps)
        pa, pb = ps[0], ps[1]
        if used[pa] >= 2 or used[pb] >= 2:
            continue
        used[pa] += 1; used[pb] += 1
        jobs.append(("multi", [pa, pb], MULTI.format(concept=c), f"[A]\n{fmt(byid[pa])}\n\n[B]\n{fmt(byid[pb])}", c))
        if sum(j[0] == "multi" for j in jobs) >= a.n_multi * 2:
            break

    def work(job):
        typ, gold, sys_p, user_p, concept = job
        text, usage = chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
                           temperature=0.7, run_id=a.seed, json_mode=False)
        d = parse_json(text) or {}
        return {"type": typ, "gold_ids": gold, "concept": concept, "question": str(d.get("question", "")).strip(),
                "reference_answer": str(d.get("answer", "")).strip(), "usage": usage}
    res, errors = run_parallel(work, jobs, workers=a.workers, desc="make_qa")
    if errors:
        print(f"{len(errors)} lỗi, ví dụ: {errors[0]}")
    rows = {"single": [], "multi": []}
    for r in res:
        if not r or len(r["question"]) < 15:
            continue
        if r["type"] == "single" and lcs_run(r["question"], byid[r["gold_ids"][0]]["title"]) >= 4:
            continue                                    # rò rỉ tiêu đề
        rows[r["type"]].append(r)
    keep = rows["single"][:a.n_single] + rows["multi"][:a.n_multi]
    EVAL.mkdir(parents=True, exist_ok=True)
    with open(EVAL / "qa_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "keep", "question", "reference_answer", "gold_ids", "concept", "gold_titles"])
        for i, r in enumerate(keep, 1):
            w.writerow([f"q{i:03d}", r["type"], 1, r["question"], r["reference_answer"], ";".join(r["gold_ids"]),
                        r["concept"] or "", " || ".join(byid[g]["title"] for g in r["gold_ids"])])
    print(f"Đã ghi {len(keep)} câu ({len(rows['single'][:a.n_single])} single, {len(rows['multi'][:a.n_multi])} multi) "
          f"vào {EVAL / 'qa_review.csv'}\nMở bằng Excel, rà tay: keep=0 để loại, sửa cột question nếu cần, rồi chạy finalize.")
    print("Chi phí:", cost_summary([r["usage"] for r in res if r]))


def finalize(a):
    out = []
    for r in csv.DictReader(open(EVAL / "qa_review.csv", encoding="utf-8-sig")):
        if str(r["keep"]).strip().lower() in ("0", "n", "no", "false"):
            continue
        out.append({"id": r["id"], "type": r["type"], "question": r["question"].strip(),
                    "reference_answer": r["reference_answer"].strip(), "gold_ids": r["gold_ids"].split(";")})
    with open(EVAL / "qa.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    n = {t: sum(o["type"] == t for o in out) for t in ("single", "multi")}
    print(f"qa.jsonl: {len(out)} câu {n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate")
    g.add_argument("--papers", default=str(ROOT / "data/processed/pilot_papers.jsonl"))
    g.add_argument("--triples", required=True)
    g.add_argument("--n-single", type=int, default=60)
    g.add_argument("--n-multi", type=int, default=40)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=6)
    sp.add_parser("finalize")
    a = ap.parse_args()
    gen(a) if a.cmd == "generate" else finalize(a)
