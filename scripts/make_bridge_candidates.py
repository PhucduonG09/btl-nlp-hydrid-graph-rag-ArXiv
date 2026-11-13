"""Cặp bridge ứng viên CÓ HƯỚNG, không dùng KG (tránh circularity):
   B 'giới thiệu' khái niệm t (t xuất hiện trong TIÊU ĐỀ của B), A 'dùng' t (t trong abstract A, KHÔNG trong tiêu đề A).
   Câu hỏi bridge sẽ mô tả A, đáp án cần B qua t; t không được xuất hiện trong câu hỏi.
  python scripts/make_bridge_candidates.py --slice data/processed/slice_c1_3000.jsonl --max-df 15
Đầu ra: results/analysis/bridge_candidates.csv"""
import sys, json, csv, re, argparse, collections, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from pick_slice import TERM, GENERIC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default=str(ROOT / "data/processed/slice_c1_3000.jsonl"))
    ap.add_argument("--max-df", type=int, default=15); ap.add_argument("--max-per-b", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default=str(ROOT / "results/analysis/bridge_candidates.csv"))
    ap.add_argument("--intro-only", action="store_true", help="chỉ giữ B_introduces=1 (tiêu đề B dạng 'Tên: mô tả')"); ap.add_argument("--extra", nargs="*", default=[], help="thêm file jsonl vào pool")
    a = ap.parse_args()
    P, seen = [], set()
    for fn in [a.slice] + a.extra:
        for l in open(fn, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["paper_id"] not in seen: seen.add(d["paper_id"]); P.append(d)
    ab = {p["paper_id"]: p["abstract"] for p in P}
    ttl = {p["paper_id"]: p["title"] for p in P}
    toks = lambda s: {t for t in TERM.findall(s) if t not in GENERIC and len(t) > 2 and not re.fullmatch(r"\d+[A-Za-z]?", t)}
    in_ab = {p["paper_id"]: toks(f"{p['title']}. {p['abstract']}") for p in P}
    df = collections.Counter(t for s in in_ab.values() for t in s)
    rows = []
    inv = collections.defaultdict(list)                       # chỉ mục ngược: term -> các A dùng term (có trong abstract, không trong tiêu đề)
    for p in P:
        tt = toks(p["title"])
        for t in toks(p["abstract"]):
            if t not in tt: inv[t].append(p["paper_id"])
    pid = {p["paper_id"]: p for p in P}
    for b in P:
        for t in toks(b["title"]):
            if not (2 <= df[t] <= a.max_df): continue
            intro = int(bool(re.match(r"\W*" + re.escape(t) + r"\s*[:\-–—]", b["title"])))
            if a.intro_only and not intro: continue
            for ida in inv.get(t, []):
                if ida == b["paper_id"]: continue
                pa = pid[ida]
                rows.append({"A_id": ida, "A_title": pa["title"], "B_id": b["paper_id"], "B_title": b["title"],
                             "bridge_term": t, "df": df[t], "B_introduces": intro})
    rng = random.Random(a.seed); rng.shuffle(rows); rows.sort(key=lambda r: -r["B_introduces"])   # ứng viên "B giới thiệu t" lên trước
    usedA, perB, keep = set(), collections.Counter(), []
    for r in rows:                                   # đa dạng: mỗi A một lần, mỗi B tối đa max-per-b
        if r["A_id"] in usedA or perB[r["B_id"]] >= a.max_per_b: continue
        usedA.add(r["A_id"]); perB[r["B_id"]] += 1; keep.append(r)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(keep[0])); w.writeheader(); w.writerows(keep)
    print(f"cặp có hướng (thô): {len(rows)} | sau khử trùng (mỗi A 1 lần, mỗi B ≤{a.max_per_b}): {len(keep)} | B duy nhất: {len(perB)}")
    print(f"trong đó tiêu đề B dạng 'Tên: mô tả' (tin cậy cao): {sum(r['B_introduces'] for r in keep)}")
    print("thuật ngữ cầu nối phổ biến nhất:", collections.Counter(r['bridge_term'] for r in keep).most_common(12))
    for r in keep[:10]: print(f"  [{r['bridge_term']} df={r['df']} intro={r['B_introduces']}] A: {r['A_title'][:70]!r} → B: {r['B_title'][:70]!r}")


if __name__ == "__main__":
    main()
