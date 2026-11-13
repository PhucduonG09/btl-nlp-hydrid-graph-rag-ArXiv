"""Baseline truy xuất hai bước KHÔNG graph, so với Graph MENTIONS đã khóa. Không API, không qa_test.
Chạy (env_AI, cần BGE local):
  python -X utf8 scripts/eval_two_hop_baseline.py --run-id twohop_01
Hai bộ: DEV v2 (122 câu: kiểm tra có hại single không) và 7 câu bridge khó (thăm dò, n nhỏ).
Biến thể khai báo trước trong src/retrieval/two_hop.py; primary = tw3_full. Không tune trên test."""
import sys, json, argparse, time, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.retrieval.two_hop import expansion_query, VARIANTS, PRIMARY
from src.retrieval.graph import fuse_rrf
from scripts.run_baselines import PREFIX
SETS = {"dev": ROOT / "data/eval/benchmark_v2_assisted/qa_dev.jsonl", "hard7": ROOT / "data/eval/qa_bridge_hard.jsonl"}
BASE = ["bm25", "dense", "rrf", "graph", "text_expansion"]


def load(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
def rank_of(lst, pid): return lst.index(pid) + 1 if pid in lst else None


def two_hop_rankings(engine, q, base):
    """Trả về {variant: ranking}. Seeds = top-3 của hop 1 (RRF BM25+dense); không graph."""
    seeds = base[:3]; lists = {}
    texts = {(i, m): expansion_query(q["question"], engine.papers[s], m) for i, s in enumerate(seeds) for m in ("full", "title")}
    keys = list(texts)
    vecs = engine.encoder.encode([PREFIX + texts[k] for k in keys], normalize_embeddings=True, show_progress_bar=False, batch_size=8)
    for k, v in zip(keys, vecs):
        dense = [engine.ids[i] for i in np.argsort(-(engine.embeddings @ v), kind="stable")[:100]]
        lists[k] = [engine.bm25.rank(texts[k], k1=1.5, b=.75), dense]
    out = {}
    for name, (n, mode) in VARIANTS.items():
        exp = [l for i in range(n) for l in lists[(i, mode)]]
        out[name] = fuse_rrf([base] + exp, k=60)
    return out


def summarize(rows, systems):
    out = {}
    for grp in ("single", "bridge"):
        sel = [r for r in rows if r["type"] == grp]
        if not sel: continue
        d = {}
        for s in systems:
            if grp == "single":
                d[s] = dict(n=len(sel), R1=float(np.mean([r["ranks"][s][0] == 1 for r in sel])),
                            R10=float(np.mean([r["ranks"][s][0] is not None and r["ranks"][s][0] <= 10 for r in sel])),
                            MRR10=float(np.mean([1 / r["ranks"][s][0] if r["ranks"][s][0] and r["ranks"][s][0] <= 10 else 0 for r in sel])))
            else:
                ok = lambda x: x is not None and x <= 10
                d[s] = dict(n=len(sel), All10=float(np.mean([ok(r["ranks"][s][0]) and ok(r["ranks"][s][1]) for r in sel])),
                            Hop1_10=float(np.mean([ok(r["ranks"][s][0]) for r in sel])), Hop2_10=float(np.mean([ok(r["ranks"][s][1]) for r in sel])),
                            B_in_top100=int(sum(r["ranks"][s][1] is not None for r in sel)),
                            B_median_rank_if_found=float(np.median([r["ranks"][s][1] for r in sel if r["ranks"][s][1]])) if any(r["ranks"][s][1] for r in sel) else None)
        out[grp] = d
    return out


def mcnemar(rows, a, b):
    ok = lambda r, s: r["ranks"][s][0] is not None and r["ranks"][s][0] <= 10 and r["ranks"][s][1] is not None and r["ranks"][s][1] <= 10
    sel = [r for r in rows if r["type"] == "bridge"]
    w = sum(ok(r, a) and not ok(r, b) for r in sel); l = sum(ok(r, b) and not ok(r, a) for r in sel)
    return dict(a=a, b=b, a_only=w, b_only=l, p_exact=float(binomtest(min(w, l), w + l, .5).pvalue) if w + l else 1.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); a = ap.parse_args()
    out = ROOT / "results/two_hop" / a.run_id
    if out.exists(): sys.exit("run-id đã tồn tại; chọn id khác.")
    out.mkdir(parents=True)
    from src.retrieval.engine import load_engine
    engine = load_engine(); systems = BASE + list(VARIANTS); report = {}
    for name, path in SETS.items():
        if "test" in path.name or "all" in path.name: sys.exit("Không dùng test/all.")
        qa = load(path); rows = []; t0 = time.perf_counter()
        for i, q in enumerate(qa, 1):
            r = engine.retrieve(q["question"], method="all"); lists = dict(r["rankings"]); lists.update(two_hop_rankings(engine, q, lists["rrf"]))
            gold = [q["hop1_id"], q["hop2_id"]] if q["type"] == "bridge" else q["gold_ids"]
            rows.append(dict(id=q["id"], type=q["type"], seeds_graph=r["seeds"], ranks={s: [rank_of(lists[s], g) for g in gold] for s in systems}))
            if i % 10 == 0 or i == len(qa): print(f"{name} {i}/{len(qa)}", flush=True)
        (out / f"{name}_per_question.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        report[name] = dict(qa=str(path.relative_to(ROOT)), qa_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), seconds=time.perf_counter() - t0,
                            metrics=summarize(rows, systems),
                            mcnemar=[mcnemar(rows, v, o) for v in VARIANTS for o in ("graph", "rrf")])
    report["primary"] = PRIMARY; report["variants"] = VARIANTS
    report["note"] = "Thăm dò/chẩn đoán. Không tune trên test. hard7 viết sau khi thấy kết quả DEV."
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in SETS:
        print(f"\n== {name} ==")
        for grp, d in report[name]["metrics"].items():
            print(grp)
            for s, v in d.items(): print(f"  {s:15s}", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != 'n'})
        for m in report[name]["mcnemar"]: print("  McNemar", m)


if __name__ == "__main__": main()
