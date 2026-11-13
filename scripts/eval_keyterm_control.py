"""Control KHÔNG graph #2: mở rộng bằng từ hiếm (idf cao) của seed, chỉ BM25. Không API, không qa_test.
Giả thuyết khai báo trước: nếu Graph MENTIONS chỉ khai thác trùng từ/tên hiếm giữa A và B thì control này
cho kết quả gần Graph trên bridge DEV (tức bridge DEV chủ yếu đo trùng từ hiếm, đúng lo ngại vòng tròn).
Chạy: python -X utf8 scripts/eval_keyterm_control.py --run-id keyterm_01"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.eval_two_hop_baseline import SETS, load, rank_of, summarize, mcnemar
from scripts.run_baselines import tok
from src.retrieval.graph import fuse_rrf
VARIANTS = {"kt3_m5": (3, 5, 1), "kt3_m10": (3, 10, 1), "kt1_m10": (1, 10, 1)}  # (số seed, số từ hiếm, df tối thiểu)
VARIANTS_DF2 = {"kt3_m5_df2": (3, 5, 2), "kt3_m10_df2": (3, 10, 2), "kt1_m10_df2": (1, 10, 2)}  # chỉ từ xuất hiện ở >=2 paper, giống điều kiện entity của Graph


def rare_terms(engine, paper, question, m, min_df=1):
    q = set(tok(question)); seen = []
    for t in tok(paper["title"] + ". " + paper["abstract"]):
        if t not in seen and t in engine.bm25.idf and t not in q and len(engine.bm25.postings[t][0]) >= min_df: seen.append(t)
    return sorted(seen, key=lambda t: (-engine.bm25.idf[t], t))[:m]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); ap.add_argument("--df2", action="store_true"); a = ap.parse_args()
    global VARIANTS
    if a.df2: VARIANTS = VARIANTS_DF2
    out = ROOT / "results/keyterm" / a.run_id
    if out.exists(): sys.exit("run-id đã tồn tại.")
    out.mkdir(parents=True)
    from src.retrieval.engine import load_engine
    engine = load_engine(); systems = ["rrf", "graph"] + list(VARIANTS); report = {}
    for name, path in SETS.items():
        qa = load(path); rows = []
        for q in qa:
            r = engine.retrieve(q["question"], method="all"); lists = dict(r["rankings"]); base = lists["rrf"]
            for v, (n, m, d) in VARIANTS.items():
                exp = [engine.bm25.rank(q["question"] + " " + " ".join(rare_terms(engine, engine.papers[s], q["question"], m, d))) for s in base[:n]]
                lists[v] = fuse_rrf([base] + exp, k=60)
            gold = [q["hop1_id"], q["hop2_id"]] if q["type"] == "bridge" else q["gold_ids"]
            rows.append(dict(id=q["id"], type=q["type"], ranks={s: [rank_of(lists[s], g) for g in gold] for s in systems}))
        report[name] = dict(metrics=summarize(rows, systems), mcnemar=[mcnemar(rows, v, o) for v in VARIANTS for o in ("graph", "rrf")])
        print(f"\n== {name} ==")
        for grp, d in report[name]["metrics"].items():
            print(grp)
            for s, v in d.items(): print(f"  {s:10s}", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != 'n'})
        for mm in report[name]["mcnemar"]: print("  McNemar", mm)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
