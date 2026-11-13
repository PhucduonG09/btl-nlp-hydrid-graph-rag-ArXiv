"""Thăm dò mức cặp (không cần câu hỏi): cho seed A, hạng của B trong
  - graph MENTIONS (đã khóa) | graph hợp nhất MENTIONS+cụm top10/top20 | dense cos(A) | BM25(title+abstract của A) | RRF(dense_A, graph).
Không API, không qa_test, không dùng test để chọn. Tập: cặp khái niệm (LLM-verified, CHƯA duyệt người), bridge tên riêng DEV.
Quy tắc khai báo trước: hợp nhất 'có giá trị' nếu Hit@10 trên cặp khái niệm cao hơn MENTIONS và dense_A, mà Hit@10 trên bridge DEV không giảm quá 3 điểm.
Chạy: python -X utf8 scripts/phrase_expansion_rank.py"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.kg.phrases import paper_phrases, top_salient, entity_index, candidate_phrases
from src.kg.mentions import MentionIndex
from src.retrieval.graph import GraphSearch, fuse_rrf
from src.retrieval.offline import BM25Index
from scripts.phrase_connectivity_audit import concept_pairs, load


def union_graph(M, papers, k, include_mentions=True):
    allp = paper_phrases(papers); ph = entity_index(top_salient(allp, k), 2, 200); ids = M["paper_ids"]
    pe = {p: sorted((set(M["paper_entities"][p]) if include_mentions else set()) | ph.get(p, set())) for p in ids}
    ep = {}
    for p, es in pe.items():
        for e in es: ep.setdefault(e, []).append(p)
    pt = {p["paper_id"]: candidate_phrases(p["title"]) for p in papers}
    te = {p: sorted((set(M["title_entities"].get(p, [])) if include_mentions else set()) | (pt[p] & set(pe[p]))) for p in ids}
    return dict(paper_ids=ids, paper_entities=pe, entity_papers=ep, title_entities=te)


def stats(ranks):
    n = len(ranks); f = [r for r in ranks if r]
    return dict(n=n, hit10=sum(r is not None and r <= 10 for r in ranks), hit100=len(f), median_rank_if_found=float(np.median(f)) if f else None,
                mrr100=round(float(np.mean([1 / r if r else 0 for r in ranks])), 3))


def main():
    manifest = json.loads((ROOT / "data/eval/benchmark_v2_assisted/manifest.json").read_text(encoding="utf-8"))
    papers = load(ROOT / manifest["corpus"].replace("\\", "/")); ids = [p["paper_id"] for p in papers]; pid2 = {p["paper_id"]: p for p in papers}
    M = json.load(open(ROOT / "data/kg/reference_mentions_v2_assisted.json", encoding="utf-8"))
    graphs = {"mentions": GraphSearch(MentionIndex(M)), "union_top10": GraphSearch(MentionIndex(union_graph(M, papers, 10))),
              "union_top20": GraphSearch(MentionIndex(union_graph(M, papers, 20))),
              "phrase_top10": GraphSearch(MentionIndex(union_graph(M, papers, 10, include_mentions=False)))}
    source = load(ROOT / "data/processed/pilot_papers.jsonl") + load(ROOT / "data/processed/distractors_20000.jsonl")
    sid = [p["paper_id"] for p in source]; key = hashlib.sha1(("|".join(sid) + "BAAI/bge-base-en-v1.5").encode()).hexdigest()[:10]
    cached = np.load(ROOT / "data/index" / f"emb_scale_{key}.npy", mmap_mode="r"); look = {p: i for i, p in enumerate(sid)}
    E = np.asarray(cached[[look[p] for p in ids]], dtype=np.float32); pos = {p: i for i, p in enumerate(ids)}
    bm25 = BM25Index(papers)
    sets = {"concept_llm_verified(30)": concept_pairs(),
            "named_bridge_dev(31)": [(x["hop1_id"], x["hop2_id"]) for x in load(ROOT / "data/eval/benchmark_v2_assisted/qa_dev.jsonl") if x["type"] == "bridge"]}
    rank = lambda lst, b: lst.index(b) + 1 if b in lst else None
    out = {}
    for sname, pairs in sets.items():
        R = {m: [] for m in ["mentions", "union_top10", "union_top20", "phrase_top10", "rrf2(mentions,phrase10)", "dense_A", "bm25_A", "rrf(dense_A+union_top10)", "rrf(dense_A+rrf2)"]}
        for a, b in pairs:
            ex = {g: [x for x in graphs[g].expand([a], hops=1)[0] if x != a][:100] for g in graphs}
            dense = [ids[i] for i in np.argsort(-(E @ E[pos[a]]), kind="stable")[:101] if ids[i] != a][:100]
            bm = [x for x in bm25.rank(pid2[a]["title"] + ". " + pid2[a]["abstract"], limit=101) if x != a][:100]
            for g in graphs: R[g].append(rank(ex[g], b))
            R["dense_A"].append(rank(dense, b)); R["bm25_A"].append(rank(bm, b))
            f2 = fuse_rrf([ex["mentions"], ex["phrase_top10"]], k=60, weights=[2, 1]); R["rrf2(mentions,phrase10)"].append(rank(f2, b))
            R["rrf(dense_A+union_top10)"].append(rank(fuse_rrf([dense, ex["union_top10"]], k=60), b)); R["rrf(dense_A+rrf2)"].append(rank(fuse_rrf([dense, f2], k=60), b))
        out[sname] = {m: stats(r) for m, r in R.items()}
        print(f"\n== {sname} ==")
        for m, v in out[sname].items(): print(f"  {m:28s} Hit@10 {v['hit10']:>2}/{v['n']}  Hit@100 {v['hit100']:>2}  MRR@100 {v['mrr100']:.3f}  median rank {v['median_rank_if_found']}")
    p = ROOT / "results/analysis/phrase_expansion_rank.json"; p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
