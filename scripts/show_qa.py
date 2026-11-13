"""Xuất data/eval/qa_review_view.txt: mỗi câu hỏi kèm abstract của paper gold để rà nhanh.
  python scripts/show_qa.py"""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
papers = {}
for l in open(ROOT / "data/processed/pilot_papers.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l); papers[d["paper_id"]] = d
rows = list(csv.DictReader(open(ROOT / "data/eval/qa_review.csv", encoding="utf-8-sig")))
rows.sort(key=lambda r: (r["type"] != "multi", r["id"]))          # câu multi lên trước
with open(ROOT / "data/eval/qa_review_view.txt", "w", encoding="utf-8") as f:
    for r in rows:
        f.write("=" * 90 + f"\n{r['id']}  [{r['type']}]  keep={r['keep']}\nCÂU HỎI: {r['question']}\nĐÁP ÁN NHÁP: {r['reference_answer']}\n")
        if r["concept"]:
            f.write(f"KHÁI NIỆM NỐI: {r['concept']}\n")
        for g in r["gold_ids"].split(";"):
            p = papers[g]
            f.write(f"--- GOLD {g}: {p['title']}\n    {p['abstract']}\n")
print(f"Đã ghi {len(rows)} câu vào data/eval/qa_review_view.txt")
