"""Tính metric rerank từ điểm đã chấm (GPU/Colab hoặc cache cũ). Offline, không API, không test.

  python scripts/rerank_bge_eval.py --scores results/benchmark_v2_assisted/dev_01/rerank_bge/scores.jsonl
Hồi quy: chạy trên điểm MiniLM cũ phải tái tạo đúng metrics.json cũ:
  python scripts/rerank_bge_eval.py --scores results/benchmark_v2_assisted/dev_01/rerank/scores.jsonl --label minilm_regress --allow-legacy-keys
Mọi biến thể đều là KẾT QUẢ DEV, thăm dò (test v2 đã mở). Việc "chọn" biến thể tốt nhất trên DEV làm số lạc quan: báo cáo cả bảng.
"""
import argparse, json, sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval.rerank import reorder_candidates
from src.retrieval.graph import fuse_rrf
from src.eval.benchmark_metrics import summarize, selection_score

MIX_WEIGHTS = [.25, .5, 1, 2]
DEPTH = 50


def load(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def mcnemar_exact(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def build_variants(originals, scores):
    """originals: {'rrf': [...], 'graph': [...]}; scores: {pid: float}. Cùng công thức run_rerank_dev.py."""
    variants = {}
    for family, ranked in originals.items():
        variants[family + "_base"] = ranked
        ce = reorder_candidates(ranked, scores, depth=DEPTH)
        variants[family + "_ce"] = ce
        for w in MIX_WEIGHTS:
            variants[family + f"_ce_mix_w{w}"] = fuse_rrf([ranked, ce], weights=[1, w])
    return variants


def success(q, ranking):
    """single: Hit@1; bridge: cả hai hop trong top-10."""
    if q["type"] == "bridge":
        top = ranking[:10]
        return q["hop1_id"] in top and q["hop2_id"] in top
    return bool(ranking) and ranking[0] in q["gold_ids"]


def paired(qa, rankings, a, b, qtype):
    sel = [q for q in qa if q["type"] == qtype]
    a_only = sum(success(q, rankings[a][q["id"]]) and not success(q, rankings[b][q["id"]]) for q in sel)
    b_only = sum(success(q, rankings[b][q["id"]]) and not success(q, rankings[a][q["id"]]) for q in sel)
    return dict(n=len(sel), a=a, b=b, a_only=a_only, b_only=b_only, p_exact=mcnemar_exact(a_only, b_only),
                metric="All@10 (hai hop trong top-10)" if qtype == "bridge" else "Hit@1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--label", default="bge_v2m3")
    ap.add_argument("--allow-legacy-keys", action="store_true", help="bỏ kiểm tra khóa (chỉ để chạy hồi quy trên điểm MiniLM cũ)")
    a = ap.parse_args()
    base = ROOT / "results/benchmark_v2_assisted/dev_01"
    out = base / "rerank_bge"
    out.mkdir(exist_ok=True)
    qa = load(ROOT / "data/eval/benchmark_v2_assisted/qa_dev.jsonl")
    source = {r["id"]: r for r in load(base / "retrieval.jsonl")}
    selected = json.loads((base / "selected_config.json").read_text(encoding="utf-8"))
    inputs = {r["id"]: r for r in load(out / "colab_input.jsonl")}
    scored = {}
    for r in load(a.scores):
        scored[r["id"]] = r  # bản ghi cuối thắng
    assert set(scored) == {q["id"] for q in qa}, f"thiếu điểm cho {len({q['id'] for q in qa} - set(scored))} câu"
    rankings, records = {}, []
    for q in qa:
        row, rec = inputs[q["id"]], scored[q["id"]]
        if not a.allow_legacy_keys:
            assert rec["key"] == row["key"], f"{q['id']}: khóa điểm không khớp đầu vào đã xuất"
        pool = [c["id"] for c in row["pool"]]
        assert set(rec["scores"]) == set(pool), f"{q['id']}: tập điểm khác pool"
        originals = dict(rrf=source[q["id"]]["rankings"][selected["rrf"]], graph=source[q["id"]]["rankings"][selected["graph"]])
        variants = build_variants(originals, rec["scores"])
        for name, ranking in variants.items():
            rankings.setdefault(name, {})[q["id"]] = ranking
        records.append(dict(id=q["id"], rankings=variants))
    metrics = {name: summarize(qa, r) for name, r in rankings.items()}
    best = {f: sorted([m for m in metrics if m.startswith(f + "_")],
                      key=lambda m: (-selection_score(metrics[m])[0], -selection_score(metrics[m])[1], m))[0] for f in ["rrf", "graph"]}
    tests = []
    for fam in ["rrf", "graph"]:
        for name in [m for m in metrics if m.startswith(fam + "_") and not m.endswith("_base")]:
            for qtype in ["bridge", "single"]:
                tests.append(paired(qa, rankings, fam + "_base", name, qtype))
    (out / f"metrics_{a.label}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / f"selected_config_{a.label}.json").write_text(json.dumps(dict(selected=best, test_evaluated=False, scores_file=str(a.scores), exploratory=True), indent=2), encoding="utf-8")
    (out / f"paired_tests_{a.label}.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
    (out / f"retrieval_{a.label}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"{'variant':24s} {'S R@1':>6s} {'S MRR':>6s} {'S R@10':>6s} {'B All@10':>8s}")
    for name, m in metrics.items():
        print(f"{name:24s} {m['single']['R@1']:6.3f} {m['single']['MRR@10']:6.3f} {m['single']['R@10']:6.3f} {m['bridge']['All@10']:8.3f}")
    print("\nMcNemar (a=base, b=biến thể):")
    for t in tests:
        print(f"  {t['b']:22s} {t['metric'][:8]:8s} base-only={t['a_only']:3d} variant-only={t['b_only']:3d} p={t['p_exact']:.3f}")
    print("Chọn theo DEV (lạc quan, chỉ tham khảo):", best)


if __name__ == "__main__":
    main()
