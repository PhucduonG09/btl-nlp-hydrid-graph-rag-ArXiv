"""Kiểm tra câu bridge có quá dễ không: chỉ dùng BM25 trên câu hỏi, B (hop2) có vào top-k không?
Corpus = slice 3000 paper (đúng corpus sẽ dùng để đánh giá).  python scripts/bridge_triviality.py [--qa file] [--k 10]
Đọc: B_rank nhỏ = hop 2 không cần đi qua A (câu yếu như bằng chứng multi-hop). A_rank nhỏ = hop 1 tìm được."""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_baselines import tok


def load_bm25(corpus):
    from rank_bm25 import BM25Okapi
    files = corpus if isinstance(corpus, (list, tuple)) else [corpus]; P, seen = [], set()
    for fn in files:
        for l in open(fn, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["paper_id"] not in seen: seen.add(d["paper_id"]); P.append(d)
    ids = [p["paper_id"] for p in P]
    return ids, BM25Okapi([tok(f"{p['title']}. {p['abstract']}") for p in P])


def ranks(ids, bm25, q, targets):
    order = [ids[i] for i in np.argsort(-bm25.get_scores(tok(q)))]
    pos = {d: r for r, d in enumerate(order, 1)}
    return [pos.get(t) for t in targets]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--qa", default=str(ROOT / "data/eval/qa_bridge_hand.jsonl"))
    ap.add_argument("--corpus", nargs="+", default=[str(ROOT / "data/processed/slice_c1_3000.jsonl")]); ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args(); ids, bm25 = load_bm25(a.corpus)
    qa = [json.loads(l) for l in open(a.qa, encoding="utf-8") if l.strip()]; nA = nB = nBoth = 0
    print(f"{'id':7}{'A_rank':>8}{'B_rank':>8}  ghi chú")
    for q in qa:
        ra, rb = ranks(ids, bm25, q["question"], [q["hop1_id"], q["hop2_id"]])
        ina, inb = ra is not None and ra <= a.k, rb is not None and rb <= a.k
        nA += ina; nB += inb; nBoth += ina and inb
        note = "B tìm thẳng được (dễ)" if inb else ("" if ina else "cả A lẫn B đều khó")
        print(f"{q['id']:7}{ra!s:>8}{rb!s:>8}  {note}")
    n = max(len(qa), 1)
    print(f"\nn={len(qa)}  A@{a.k}={nA/n:.0%}  B@{a.k}={nB/n:.0%}  cả hai={nBoth/n:.0%}")
    print("B@k cao nghĩa là câu hỏi mô tả B quá rõ; muốn bridge thật thì B@k (chỉ từ câu hỏi) nên thấp hơn nhiều so với B@k sau khi đã biết A.")


if __name__ == "__main__":
    main()
