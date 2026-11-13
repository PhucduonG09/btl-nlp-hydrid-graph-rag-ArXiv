"""Lọc snapshot ArXiv (Kaggle: Cornell-University/arxiv, file arxiv-metadata-oai-snapshot.json)
thành corpus distractor cùng schema với pilot_papers.jsonl.
  python scripts/prepare_corpus.py --snapshot D:/data/arxiv-metadata-oai-snapshot.json --n 20000
Đầu ra: data/processed/distractors_20000.jsonl (không chứa paper pilot)."""
import json, argparse, random, re
from email.utils import parsedate_to_datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CATS = {"cs.CL", "cs.LG", "cs.AI"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--pilot", default=str(ROOT / "data/processed/pilot_papers.jsonl"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-abs-words", type=int, default=50)
    a = ap.parse_args()

    pilot = [json.loads(l) for l in open(a.pilot, encoding="utf-8") if l.strip()]
    pilot_ids = {p["paper_id"] for p in pilot}
    dates = sorted(p["date"] for p in pilot if p.get("date"))
    lo, hi = dates[0], dates[-1]  # cùng khoảng thời gian với pilot
    print(f"pilot: {len(pilot)} paper, date {lo}..{hi}")

    cand = []
    with open(a.snapshot, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cats = r.get("categories", "").split()
            if not CATS.intersection(cats) or r["id"] in pilot_ids:
                continue
            date = r["versions"][0]["created"] if r.get("versions") else r.get("update_date")
            try:  # "Mon, 6 Jan 2025 10:00:00 GMT" (ngày có thể không đệm số 0)
                d = parsedate_to_datetime(date).date().isoformat()
            except Exception:
                d = (date or "")[:10]
            abstract = re.sub(r"\s+", " ", r.get("abstract", "")).strip()
            if not (lo <= d <= hi) or len(abstract.split()) < a.min_abs_words:
                continue
            cand.append({"paper_id": r["id"], "title": re.sub(r"\s+", " ", r["title"]).strip(),
                         "abstract": abstract, "categories": cats, "date": d,
                         "authors": [" ".join(x[:2][::-1]).strip() for x in r.get("authors_parsed", [])]})
    print(f"ứng viên sau lọc: {len(cand)}")
    random.Random(a.seed).shuffle(cand)
    out = ROOT / f"data/processed/distractors_{a.n}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in cand[:a.n]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"ghi {min(a.n, len(cand))} paper -> {out}")


if __name__ == "__main__":
    main()
