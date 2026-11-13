"""Đánh giá MỨC CẶP với ground truth ĐỘC LẬP từ ngữ: A trích dẫn B (cả hai trong corpus v2). Không API, không qa_test.
Hạng của B khi chỉ biết A, cho: graph MENTIONS, graph cụm top10, rrf2(mentions,phrase10), dense cos(A), BM25(A), RRF(dense_A,bm25_A) [hybrid không graph],
RRF(dense_A, mentions), RRF(dense_A, rrf2).  Chia tầng: cặp có/không có chung thực thể MENTIONS (dễ theo từ vựng vs không).
Giả thuyết khai báo trước (chưa biết kết quả): (H-a) dense_A mạnh trên cặp trích dẫn vì trích dẫn gần chủ đề; (H-b) graph MENTIONS yếu vì phần lớn cặp không chung tên riêng;
(H-c) hợp nhất dense_A với graph cụm bổ sung cho nhau (Hit@10 của RRF(dense_A, rrf2) > dense_A). Quyết định mở rộng graph khái niệm chỉ khi H-c đúng với khoảng tin cậy bootstrap không chứa 0.
Chạy: python -X utf8 scripts/citation_pair_rank.py [--edges data/external/citations_s2.jsonl] [--max-pairs 800]"""
import sys, json, argparse, hashlib, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.phrase_expansion_rank import union_graph
from scripts.phrase_connectivity_audit import load
from src.kg.mentions import MentionIndex
from src.retrieval.graph import GraphSearch, fuse_rrf
from src.retrieval.offline import BM25Index
SYS = ["mentions", "phrase_top10", "rrf2(mentions,phrase10)", "dense_A", "bm25_A", "rrf(dense_A,bm25_A)", "rrf(dense_A,mentions)", "rrf(dense_A,rrf2)"]


def hit(r, k=10): return r is not None and r <= k


def boot_diff(a, b, n=5000, seed=0):
    d = np.array([float(x) - float(y) for x, y in zip(a, b)]); rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--edges", default="data/external/citations_s2.jsonl"); ap.add_argument("--max-pairs", type=int, default=800)
    ap.add_argument("--out", default="results/analysis/citation_pair_rank.json"); ap.add_argument("--dump", default=None); a = ap.parse_args()
    manifest = json.loads((ROOT / "data/eval/benchmark_v2_assisted/manifest.json").read_text(encoding="utf-8"))
    papers = load(ROOT / manifest["corpus"].replace("\\", "/")); ids = [p["paper_id"] for p in papers]; pid2 = {p["paper_id"]: p for p in papers}
    M = json.load(open(ROOT / "data/kg/reference_mentions_v2_assisted.json", encoding="utf-8"))
    edges = sorted({tuple(e) for l in Path(ROOT / a.edges).read_text(encoding="utf-8").splitlines() if l.strip() for e in json.loads(l)["edges"]})
    print("cạnh trích dẫn trong corpus:", len(edges), " paper có >=1 cạnh ra:", len({e[0] for e in edges}), flush=True)
    random.seed(0); pairs = random.sample(edges, min(a.max_pairs, len(edges)))
    G = {"mentions": GraphSearch(MentionIndex(M)), "phrase_top10": GraphSearch(MentionIndex(union_graph(M, papers, 10, include_mentions=False)))}
    source = load(ROOT / "data/processed/pilot_papers.jsonl") + load(ROOT / "data/processed/distractors_20000.jsonl")
    sid = [p["paper_id"] for p in source]; key = hashlib.sha1(("|".join(sid) + "BAAI/bge-base-en-v1.5").encode()).hexdigest()[:10]
    cached = np.load(ROOT / "data/index" / f"emb_scale_{key}.npy", mmap_mode="r"); look = {p: i for i, p in enumerate(sid)}
    E = np.asarray(cached[[look[p] for p in ids]], dtype=np.float32); pos = {p: i for i, p in enumerate(ids)}; bm25 = BM25Index(papers)
    pe = M["paper_entities"]; rank = lambda lst, b: lst.index(b) + 1 if b in lst else None
    R = {s: [] for s in SYS}; lexical = []; dump = []
    df = {e: len(v) for e, v in M["entity_papers"].items()}
    for n, (x, b) in enumerate(pairs, 1):
        ex = {g: [y for y in G[g].expand([x], hops=1)[0] if y != x][:100] for g in G}
        dense = [ids[i] for i in np.argsort(-(E @ E[pos[x]]), kind="stable")[:101] if ids[i] != x][:100]
        bm = [y for y in bm25.rank(pid2[x]["title"] + ". " + pid2[x]["abstract"], limit=101) if y != x][:100]
        f2 = fuse_rrf([ex["mentions"], ex["phrase_top10"]], k=60, weights=[2, 1])
        out = {"mentions": ex["mentions"], "phrase_top10": ex["phrase_top10"], "rrf2(mentions,phrase10)": f2, "dense_A": dense, "bm25_A": bm,
               "rrf(dense_A,bm25_A)": fuse_rrf([dense, bm], k=60), "rrf(dense_A,mentions)": fuse_rrf([dense, ex["mentions"]], k=60), "rrf(dense_A,rrf2)": fuse_rrf([dense, f2], k=60)}
        for s in SYS: R[s].append(rank(out[s], b))
        lexical.append(bool(set(pe[x]) & set(pe[b])))
        top = ex["mentions"][0] if ex["mentions"] else None
        shared = set(pe[x]) & set(pe[top]) if top else set()
        dump.append(dict(A=x, B=b, mentions=ex["mentions"], dense=dense, bm25=bm, f2=f2, gate_min_df=min((df[e] for e in shared), default=None)))
        if n % 100 == 0: print(f"{n}/{len(pairs)}", flush=True)
    res = dict(n_edges_total=len(edges), n_pairs=len(pairs), strata={}, bootstrap={})
    for name, mask in [("all", [True] * len(pairs)), ("shares_mentions_entity", lexical), ("no_shared_mentions_entity", [not v for v in lexical])]:
        idx = [i for i, m in enumerate(mask) if m]; res["strata"][name] = dict(n=len(idx))
        if not idx: continue
        for s in SYS:
            r = [R[s][i] for i in idx]; res["strata"][name][s] = dict(hit10=round(float(np.mean([hit(v) for v in r])), 3), hit100=round(float(np.mean([v is not None for v in r])), 3),
                                                                      mrr100=round(float(np.mean([1 / v if v else 0 for v in r])), 3))
    for s1, s2 in [("rrf(dense_A,rrf2)", "dense_A"), ("rrf(dense_A,mentions)", "dense_A"), ("rrf(dense_A,rrf2)", "rrf(dense_A,bm25_A)"), ("dense_A", "mentions")]:
        d, ci = boot_diff([hit(v) for v in R[s1]], [hit(v) for v in R[s2]]); res["bootstrap"][f"Hit@10 {s1} - {s2}"] = dict(diff=round(d, 3), ci95=[round(c, 3) for c in ci])
    if a.dump: Path(ROOT / a.dump).write_text(json.dumps(dump), encoding="utf-8")
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True); Path(ROOT / a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, v in res["strata"].items():
        if not v["n"]: continue
        print(f"\n== {name} (n={v['n']}) =="); [print(f"  {s:26s} Hit@10 {v[s]['hit10']:.3f}  Hit@100 {v[s]['hit100']:.3f}  MRR@100 {v[s]['mrr100']:.3f}") for s in SYS]
    print("\nBootstrap:"); [print("  ", k, v) for k, v in res["bootstrap"].items()]


if __name__ == "__main__": main()
