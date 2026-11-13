"""Đường cong độ khó theo quy mô corpus: giữ 70 câu hỏi, tăng dần số paper nhiễu.
  python scripts/scale_curve.py --distractors data/processed/distractors_20000.jsonl --sizes 0 2000 5000 20000
Distractor lấy ngẫu nhiên từ cùng category/khoảng thời gian (xem hạn chế trong README kết quả)."""
import sys, json, argparse, hashlib, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.eval.ir_metrics import METRICS
from scripts.run_baselines import tok, rrf, PREFIX


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default=str(ROOT / "data/processed/pilot_papers.jsonl"))
    ap.add_argument("--distractors", required=True)
    ap.add_argument("--qa", default=str(ROOT / "data/eval/qa.jsonl"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 2000, 5000, 20000])
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()

    pilot, dis, qa = load(a.pilot), load(a.distractors), load(a.qa)
    queries = [q["question"] for q in qa]
    txt = lambda p: f"{p['title']}. {p['abstract']}"
    allp = pilot + dis                      # dis đã được shuffle sẵn, lấy prefix = mẫu ngẫu nhiên lồng nhau
    ids = [p["paper_id"] for p in allp]
    texts = [txt(p) for p in allp]

    D = Q = None
    if not a.no_dense:
        from sentence_transformers import SentenceTransformer
        import torch
        enc = SentenceTransformer(a.model, device="cuda" if torch.cuda.is_available() else "cpu")
        h = hashlib.sha1(("|".join(ids) + a.model).encode()).hexdigest()[:10]
        cache = ROOT / "data/index" / f"emb_scale_{h}.npy"
        if cache.exists():
            D = np.load(cache)
        else:
            D = enc.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True).astype("float32")
            cache.parent.mkdir(parents=True, exist_ok=True); np.save(cache, D)
        Q = enc.encode([PREFIX + q for q in queries], normalize_embeddings=True).astype("float32")

    from rank_bm25 import BM25Okapi
    toks = [tok(t) for t in texts]
    rows = []
    for n in a.sizes:
        m = len(pilot) + min(n, len(dis))
        bm = BM25Okapi(toks[:m])
        rk = {"bm25": [[ids[i] for i in np.argsort(-bm.get_scores(tok(q)))[:100]] for q in queries]}
        if D is not None:
            S = Q @ D[:m].T
            rk["dense"] = [[ids[i] for i in np.argsort(-S[j])[:100]] for j in range(len(qa))]
            rk["rrf"] = [rrf([rk["bm25"][j], rk["dense"][j]]) for j in range(len(qa))]
        for typ in ("single", "multi"):
            idx = [j for j, q in enumerate(qa) if q["type"] == typ]
            for s, rl in rk.items():
                rec = {"corpus": m, "type": typ, "system": s, "n_q": len(idx)}
                for name, fn in METRICS.items():
                    rec[name] = round(float(np.mean([fn(rl[j], qa[j]["gold_ids"]) for j in idx])), 4)
                rec["allgold@10"] = round(float(np.mean(
                    [all(g in rl[j][:10] for g in qa[j]["gold_ids"]) for j in idx])), 4)
                rows.append(rec); print(rec, flush=True)
    out = ROOT / "results/scale_curve.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("ghi", out)


if __name__ == "__main__":
    main()
