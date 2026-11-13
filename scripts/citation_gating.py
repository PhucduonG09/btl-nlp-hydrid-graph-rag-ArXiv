"""Gating graph theo đặc trưng TRIỂN KHAI ĐƯỢC (không dùng B): chỉ dùng graph khi ứng viên top-1 của graph chia sẻ với seed A một thực thể có df <= tau.
So sánh trên cặp trích dẫn (ground truth độc lập từ ngữ): base = RRF(BM25,dense, k=10) vs pipeline graph như engine (RRF base+graph, k=60, trọng số 1:2) vs bản gating.
Chọn tau bằng cross-fitting 2 fold theo hash(A) (chọn trên fold kia, đánh giá trên fold này) để tránh tune trên chính dữ liệu đánh giá.
Quy tắc thắng khai báo trước: Hit@10 gating (held-out, gộp 2 fold) - base có CI bootstrap 95% > 0.
Chạy: python -X utf8 scripts/citation_gating.py [--dump results/analysis/citation_lists_4000.json]"""
import sys, json, argparse, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.retrieval.graph import fuse_rrf
TAUS = [2, 3, 5, 10, 20, 50, 200]


def fold(a): return int(hashlib.md5(a.encode()).hexdigest(), 16) % 2
def hit(lst, b): return b in lst[:10]


def boot(d, n=5000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed); m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return round(float(d.mean()), 4), [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dump", default="results/analysis/citation_lists_4000.json"); a = ap.parse_args()
    rows = json.loads((ROOT / a.dump).read_text(encoding="utf-8")); print("pairs", len(rows))
    base_hit, graph_hit, gate_hit = [], [], {t: [] for t in TAUS}; folds = []
    for r in rows:
        base = fuse_rrf([r["bm25"], r["dense"]], k=10); graph = fuse_rrf([base, r["mentions"]], k=60, weights=[1, 2])
        bh, gh = hit(base, r["B"]), hit(graph, r["B"]); base_hit.append(bh); graph_hit.append(gh); folds.append(fold(r["A"]))
        for t in TAUS: gate_hit[t].append(gh if (r["gate_min_df"] is not None and r["gate_min_df"] <= t) else bh)
    base_hit, graph_hit, folds = np.array(base_hit, float), np.array(graph_hit, float), np.array(folds)
    gate_hit = {t: np.array(v, float) for t, v in gate_hit.items()}
    gated_share = {t: round(float(np.mean([r["gate_min_df"] is not None and r["gate_min_df"] <= t for r in rows])), 3) for t in TAUS}
    res = dict(n=len(rows), base_hit10=round(float(base_hit.mean()), 4), ungated_graph_hit10=round(float(graph_hit.mean()), 4),
               ungated_minus_base=boot(graph_hit - base_hit), per_tau={str(t): dict(hit10=round(float(gate_hit[t].mean()), 4), share_gated=gated_share[t]) for t in TAUS})
    held, chosen = np.zeros(len(rows)), {}
    for f in (0, 1):
        tr, te = folds != f, folds == f
        best = max(TAUS, key=lambda t: (gate_hit[t][tr].mean(), -t)); chosen[f] = best; held[te] = gate_hit[best][te]
    res["cross_fit_tau"] = chosen; res["gated_heldout_hit10"] = round(float(held.mean()), 4); res["gated_minus_base_heldout"] = boot(held - base_hit)
    res["gated_minus_ungated_graph_heldout"] = boot(held - graph_hit)
    (ROOT / "results/analysis/citation_gating.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
