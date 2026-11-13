"""Error analysis cho câu multi-hop trên kết quả baseline đã lưu.
Chạy từ thư mục gốc repo: python scripts/error_analysis_multihop.py
Đầu ra: in bảng + results/analysis/multihop_partial.csv
"""
import json, csv, os
QA = "data/eval/qa.jsonl"
RUNS = {m: f"results/runs/retrieval_{m}.jsonl" for m in ["bm25", "dense", "rrf"]}
KS = [1, 5, 10, 20]

qa = {r["id"]: r for r in map(json.loads, open(QA, encoding="utf-8"))}
runs = {m: {r["id"]: r["ranked"] for r in map(json.loads, open(p, encoding="utf-8"))}
        for m, p in RUNS.items()}

def rank_of(ranked, pid):
    return ranked.index(pid) + 1 if pid in ranked else None

def stats(qtype):
    ids = [i for i, r in qa.items() if r["type"] == qtype]
    print(f"\n=== {qtype} (n={len(ids)}) ===")
    print("method | " + " | ".join(f"all-gold@{k}" for k in KS) + " | " + " | ".join(f"any-gold@{k}" for k in KS))
    for m in RUNS:
        allg, anyg = [], []
        for k in KS:
            a = sum(all(g in runs[m][i][:k] for g in qa[i]["gold_ids"]) for i in ids)
            b = sum(any(g in runs[m][i][:k] for g in qa[i]["gold_ids"]) for i in ids)
            allg.append(f"{a}/{len(ids)}"); anyg.append(f"{b}/{len(ids)}")
        print(f"{m:5s} | " + " | ".join(allg) + " | " + " | ".join(anyg))

stats("single"); stats("multi")

rows = []
print("\n=== Chi tiết multi (rank của từng gold, None = ngoài top-20) ===")
for i, r in qa.items():
    if r["type"] != "multi": continue
    print(f"\n{i} gold={r['gold_ids']}\n  Q: {r['question']}")
    for m in RUNS:
        ranks = [rank_of(runs[m][i], g) for g in r["gold_ids"]]
        found = sum(x is not None and x <= 10 for x in ranks)
        print(f"  {m:5s} ranks={ranks}  found@10={found}/{len(ranks)}")
        if found == 1 and len(ranks) == 2:
            rows.append({"id": i, "method": m, "ranks": ranks, "question": r["question"]})
os.makedirs("results/analysis", exist_ok=True)
with open("results/analysis/multihop_partial.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["id", "method", "ranks", "question"])
    w.writeheader(); w.writerows(rows)
print(f"\nGhi {len(rows)} dòng 'chỉ tìm được 1/2' -> results/analysis/multihop_partial.csv")
