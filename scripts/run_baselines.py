"""Chạy BM25, Dense (bge-base), RRF trên qa.jsonl và đo Recall/MRR/nDCG theo loại câu hỏi.
  python scripts/run_baselines.py"""
import sys, re, json, hashlib, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.eval.ir_metrics import METRICS

STOP = set("a an the of and or in on for to with by from is are was were be this that we our these which as at it its".split())
PREFIX = "Represent this sentence for searching relevant passages: "


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", s.lower()) if w not in STOP]


def rrf(rank_lists, k=60, top=100):
    sc = {}
    for rl in rank_lists:
        for r, d in enumerate(rl, 1):
            sc[d] = sc.get(d, 0.0) + 1.0 / (k + r)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])][:top]


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_p(a, b, n=5000, seed=0):
    """Paired bootstrap: xác suất (hai phía) hiệu trung bình a-b có dấu ngược với quan sát."""
    rng = np.random.default_rng(seed); d = np.asarray(a, float) - np.asarray(b, float)
    ms = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(min(1.0, 2 * min((ms <= 0).mean(), (ms >= 0).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default=str(ROOT / "data/processed/pilot_papers.jsonl"))
    ap.add_argument("--qa", default=str(ROOT / "data/eval/qa.jsonl"))
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()

    papers = [json.loads(l) for l in open(a.papers, encoding="utf-8") if l.strip()]
    ids = [p["paper_id"] for p in papers]
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    qa = [json.loads(l) for l in open(a.qa, encoding="utf-8") if l.strip()]
    queries = [q["question"] for q in qa]

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tok(t) for t in texts])
    rank = {"bm25": [[ids[i] for i in np.argsort(-bm25.get_scores(tok(q)))[:100]] for q in queries]}

    if not a.no_dense:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        h = hashlib.sha1(("|".join(ids) + a.model).encode()).hexdigest()[:10]
        cache = ROOT / "data" / "index" / f"emb_{h}.npy"
        enc = SentenceTransformer(a.model, device=dev)
        if cache.exists():
            D = np.load(cache)
        else:
            D = enc.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype("float32")
            cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, D)
        Q = enc.encode([PREFIX + q for q in queries], normalize_embeddings=True, batch_size=64).astype("float32")
        S = Q @ D.T
        rank["dense"] = [[ids[i] for i in np.argsort(-S[j])[:100]] for j in range(len(queries))]
        rank["rrf"] = [rrf([rank["bm25"][j], rank["dense"][j]]) for j in range(len(queries))]

    (ROOT / "results" / "runs").mkdir(parents=True, exist_ok=True)
    for s, rl in rank.items():
        with open(ROOT / "results" / "runs" / f"retrieval_{s}.jsonl", "w", encoding="utf-8") as f:
            for q, r in zip(qa, rl):
                f.write(json.dumps({"id": q["id"], "ranked": r[:20]}) + "\n")

    per = {s: {m: np.array([fn(rl[j], qa[j]["gold_ids"]) for j in range(len(qa))]) for m, fn in METRICS.items()}
           for s, rl in rank.items()}
    result, lines = {}, []
    for typ in ("all", "single", "multi"):
        idx = [j for j, q in enumerate(qa) if typ == "all" or q["type"] == typ]
        if not idx:
            continue
        lines.append(f"\n== {typ} (n={len(idx)}) ==")
        lines.append("system  " + "  ".join(f"{m:>16}" for m in METRICS))
        for s in rank:
            cells, rec = [], {}
            for m in METRICS:
                x = per[s][m][idx]; lo, hi = boot_ci(x)
                cells.append(f"{x.mean():.3f} [{lo:.2f},{hi:.2f}]"); rec[m] = round(float(x.mean()), 4)
            result[f"{typ}/{s}"] = rec
            lines.append(f"{s:<7} " + "  ".join(f"{c:>16}" for c in cells))
        if "dense" in rank and len(idx) >= 5:
            for m in ("nDCG@10", "R@10"):
                p = paired_p(per["dense"][m][idx], per["bm25"][m][idx])
                lines.append(f"  dense vs bm25 {m}: diff={per['dense'][m][idx].mean() - per['bm25'][m][idx].mean():+.3f}, p≈{p:.3f} (paired bootstrap)")
    print("\n".join(lines))
    (ROOT / "results" / "baseline_retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nĐã ghi results/baseline_retrieval.json và results/runs/retrieval_*.jsonl")


if __name__ == "__main__":
    main()
