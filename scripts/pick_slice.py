"""Chọn lát cắt chủ đề cho KG: KMeans trên embedding đã cache (từ scale_curve.py), mô tả từng cụm,
ước lượng mật độ concept chung + số cặp bridge ứng viên. Không tốn API.
  python scripts/pick_slice.py --distractors data/processed/distractors_20000.jsonl --k 12
  python scripts/pick_slice.py --distractors ... --k 12 --pick 3 --size 3000   # ghi lát cắt
Proxy 'concept' = từ viết hoa/viết tắt (GPT-4, LoRA, ImageNet, MMLU...), xấp xỉ tên method/dataset."""
import sys, re, json, argparse, hashlib, csv, itertools, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np

TERM = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9\-]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9\-]*|[A-Za-z]+-?[0-9]+[A-Za-z]*)\b")
GENERIC = {"AI", "ML", "NLP", "LLM", "LLMs", "GPU", "CPU", "AUC", "SOTA", "IEEE", "ACM", "ICML", "NeurIPS", "ICLR"}
WORD = re.compile(r"[a-z]{4,}")
STOP = set("this that with from which their have been these using based show results paper method methods approach "
           "propose proposed model models data performance learning also such than into more most both while "
           "between over under each other novel large study work existing however further across well".split())


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def terms(p):
    return {t for t in TERM.findall(f"{p['title']}. {p['abstract']}") if t not in GENERIC and len(t) > 2}


def kmeans(X, k, seed=0):
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=2048).fit(X)
        return km.labels_, km.cluster_centers_.astype("float32")
    except ImportError:
        import faiss
        km = faiss.Kmeans(X.shape[1], k, niter=30, seed=seed, spherical=True); km.train(X)
        return km.index.search(X, 1)[1][:, 0], km.centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=str(ROOT / "data/processed/pilot_papers.jsonl"))
    ap.add_argument("--distractors", required=True)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--pick", type=int, default=None)
    ap.add_argument("--size", type=int, default=3000)
    a = ap.parse_args()

    papers = load(a.pilot) + load(a.distractors)
    ids = [p["paper_id"] for p in papers]
    h = hashlib.sha1(("|".join(ids) + a.model).encode()).hexdigest()[:10]
    cache = ROOT / "data/index" / f"emb_scale_{h}.npy"
    if not cache.exists():
        sys.exit(f"Không thấy {cache}. Chạy scale_curve.py với đúng file distractors trước (cùng thứ tự paper).")
    X = np.load(cache).astype("float32"); assert len(X) == len(papers)
    labels, C = kmeans(X, a.k)
    npilot = len(load(a.pilot))
    T = [terms(p) for p in papers]
    df_all = collections.Counter(t for s in T for t in s)

    rows = []
    for c in range(a.k):
        idx = np.where(labels == c)[0]
        df = collections.Counter(t for i in idx for t in T[i])
        shared = [t for t, d in df.items() if 2 <= d <= max(20, len(idx) // 20)]   # concept 'rare vừa' → bridge được
        pairs = set()
        for t in shared:
            ps = [i for i in idx if t in T[i]]
            if len(ps) > 40: continue
            pairs.update(itertools.combinations(ps, 2))
        w = collections.Counter(w for i in idx for w in set(WORD.findall(papers[i]["abstract"].lower())) if w not in STOP)
        allw = collections.Counter(w for p in papers for w in set(WORD.findall(p["abstract"].lower())) if w not in STOP)
        top = sorted(w, key=lambda x: -(w[x] / len(idx)) * np.log((len(papers) / allw[x]) + 1e-9) if w[x] >= 15 else 0)[:8]
        rows.append({"cluster": c, "size": len(idx), "pilot": int((idx < npilot).sum()),
                     "shared_terms": len(shared), "shared_per_paper": round(len(shared) / len(idx), 3),
                     "bridge_pairs": len(pairs), "top_words": " ".join(top)})
    print(f"baseline toàn corpus: {sum(1 for t,d in df_all.items() if d >= 2)/len(papers):.3f} term(d≥2)/paper")
    print(f"{'c':>2} {'size':>6} {'pilot':>5} {'shr/paper':>9} {'bridge_pairs':>12}  top_words")
    for r in sorted(rows, key=lambda r: -r["shared_per_paper"]):
        print(f"{r['cluster']:>2} {r['size']:>6} {r['pilot']:>5} {r['shared_per_paper']:>9} {r['bridge_pairs']:>12}  {r['top_words']}")
    out = ROOT / "results/analysis/cluster_summary.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)

    if a.pick is not None:   # lấy `size` paper gần tâm cụm nhất trên toàn corpus → kích thước cố định
        sim = X @ (C[a.pick] / np.linalg.norm(C[a.pick]))
        sel = np.argsort(-sim)[:a.size]
        dst = ROOT / f"data/processed/slice_c{a.pick}_{a.size}.jsonl"
        with open(dst, "w", encoding="utf-8") as f:
            for i in sel: f.write(json.dumps(papers[i], ensure_ascii=False) + "\n")
        print(f"ghi {len(sel)} paper (pilot trong lát cắt: {(sel < npilot).sum()}) -> {dst}")


if __name__ == "__main__":
    main()
