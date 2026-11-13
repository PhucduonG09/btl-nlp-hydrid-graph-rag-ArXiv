"""Chọn cặp (A trích dẫn B) TỰ NHIÊN cho bộ bridge dựa trên trích dẫn. KHÔNG lọc theo từ vựng/graph/dense/BM25 (chỉ ghi lại làm biến phụ).
Quy tắc khai báo trước (docs/protocol-citation-bridge.md): A mới hơn B (id arXiv); B có in-degree trong corpus <= 10 (loại tham chiếu nền/hub như ToT, Toolformer);
mỗi paper xuất hiện tối đa một lần (độc lập giữa các câu); loại cặp đã nằm trong qa_dev / bridge khó; lấy ngẫu nhiên N cặp theo seed cố định.
Chạy: python -X utf8 scripts/select_citation_pairs.py [--n 320]   (không API, không đọc qa_test)"""
import sys, json, random, argparse, hashlib, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
SEED = 20261112; MAX_B_INDEG = 10


def select(edges, exclude_papers, n, seed=SEED, max_b_indeg=MAX_B_INDEG):
    indeg = collections.Counter(b for _, b in edges)
    pool = [(a, b) for a, b in edges if a > b and indeg[b] <= max_b_indeg and a not in exclude_papers and b not in exclude_papers]
    random.Random(seed).shuffle(pool); used, out = set(), []
    for a, b in pool:
        if a in used or b in used: continue
        used.update((a, b)); out.append((a, b))
        if len(out) == n: break
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=320); a = ap.parse_args()
    out_dir = ROOT / "data/eval/citation_bridge"; out_dir.mkdir(parents=True, exist_ok=True); target = out_dir / "selected_pairs.jsonl"
    if target.exists(): sys.exit("selected_pairs.jsonl đã tồn tại; không ghi đè (đã đóng băng).")
    edges = sorted({tuple(e) for l in (ROOT / "data/external/citations_s2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() for e in json.loads(l)["edges"]})
    excl = set()
    for f in [ROOT / "data/eval/benchmark_v2_assisted/qa_dev.jsonl", ROOT / "data/eval/qa_bridge_hard.jsonl"]:
        for l in f.read_text(encoding="utf-8").splitlines():
            if l.strip(): excl.update(json.loads(l)["gold_ids"])
    pairs = select(edges, excl, a.n)
    M = json.load(open(ROOT / "data/kg/reference_mentions_v2_assisted.json", encoding="utf-8")); pe = M["paper_entities"]
    rows = [dict(pair_id=f"cb{i + 1:03d}", A_id=x, B_id=y, covariate_shares_mentions_entity=bool(set(pe[x]) & set(pe[y]))) for i, (x, y) in enumerate(pairs)]
    target.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = dict(seed=SEED, max_b_indeg=MAX_B_INDEG, n=len(rows), n_edges=len(edges), sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                share_mentions_entity=round(sum(r["covariate_shares_mentions_entity"] for r in rows) / len(rows), 3))
    (out_dir / "selection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8"); print(meta)


if __name__ == "__main__": main()
