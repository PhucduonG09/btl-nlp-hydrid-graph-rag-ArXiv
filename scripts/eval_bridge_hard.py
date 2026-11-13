"""Retrieval THĂM DÒ (exploratory) trên bộ bridge khó, dùng đúng 5 phương pháp và cấu hình ĐÃ KHÓA của DEV.
Không tune, không mở qa_test. Không gọi API (chỉ model BGE local).  Chạy ở máy có BGE:
  .venv-kg/Scripts/python.exe -X utf8 scripts/eval_bridge_hard.py --qa data/eval/qa_bridge_hard.jsonl --run-id hard_01
Đầu ra: results/bridge_hard/<run-id>/{retrieval.jsonl, metrics.json, README.md}.
Chỉ số: All@10 (đủ mọi gold paper trong top-10), Any@10, Hop@10 theo thứ tự gold_ids, 95% CI bootstrap theo câu.
Vì bộ này viết SAU khi đã thấy kết quả DEV nên chỉ dùng để thăm dò giới hạn của graph MENTIONS, không phải kết quả confirmatory."""
import sys, json, argparse, time, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
METHODS = ["bm25", "dense", "rrf", "text_expansion", "graph"]


def load(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def metrics(qa, rankings, k=10):
    out = {}
    def row(q, r):
        top = r[q["id"]][:k]; hits = [g in top for g in q["gold_ids"]]
        return dict(all=float(all(hits)), any=float(any(hits)), hops=hits)
    for name, sel in [("all", qa)] + [(t, [q for q in qa if q["type"] == t]) for t in sorted({q["type"] for q in qa})]:
        if not sel: continue
        rows = [row(q, rankings) for q in sel]; a = np.array([x["all"] for x in sel and rows])
        rng = np.random.default_rng(0); boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)]
        out[name] = dict(n=len(sel), **{f"All@{k}": float(a.mean()), f"Any@{k}": float(np.mean([x["any"] for x in rows]))},
                         **{f"AllCI95_{k}": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]},
                         **{f"Gold{j+1}@{k}": float(np.mean([x["hops"][j] for x in rows if len(x["hops"]) > j])) for j in range(max(len(q["gold_ids"]) for q in sel))})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--qa", required=True); ap.add_argument("--run-id", default="hard_01"); a = ap.parse_args()
    if "qa_test" in a.qa or "qa_all" in a.qa: sys.exit("Không dùng file test/all ở đây.")
    out = ROOT / "results/bridge_hard" / a.run_id
    if out.exists(): sys.exit("Thư mục kết quả đã tồn tại; chọn --run-id khác để giữ kết quả cũ.")
    qa = load(a.qa); out.mkdir(parents=True)
    from src.retrieval.engine import load_engine
    engine = load_engine(); ids = set(engine.papers) if hasattr(engine, "papers") else None
    if ids is not None:
        missing = [(q["id"], g) for q in qa for g in q["gold_ids"] if g not in ids]
        if missing: sys.exit(f"Gold không nằm trong corpus v2: {missing[:5]}")
    rankings = {m: {} for m in METHODS}; t0 = time.perf_counter()
    with (out / "retrieval.jsonl").open("w", encoding="utf-8") as f:
        for i, q in enumerate(qa, 1):
            r = engine.retrieve(q["question"], method="all"); f.write(json.dumps(dict(id=q["id"], **r), ensure_ascii=False) + "\n")
            for m in METHODS: rankings[m][q["id"]] = r["rankings"][m]
            print(f"{i}/{len(qa)}", flush=True)
    res = {m: metrics(qa, rankings[m]) for m in METHODS}
    (out / "metrics.json").write_text(json.dumps(dict(qa=a.qa, qa_sha256=hashlib.sha256(Path(a.qa).read_bytes()).hexdigest(), seconds=time.perf_counter() - t0,
                                                    exploratory=True, methods=res), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# Bridge khó, thăm dò ({len(qa)} câu): All@10", "", "| Phương pháp | " + " | ".join(res['rrf'].keys()) + " |", "|---|" + "---|" * len(res['rrf'])]
    for m in METHODS: lines.append(f"| {m} | " + " | ".join(f"{res[m][g]['All@10']:.2f} (n={res[m][g]['n']})" for g in res['rrf']) + " |")
    lines += ["", "Thăm dò: câu viết sau khi thấy kết quả DEV, cấu hình khóa từ DEV, không tune. Không dùng thành kết quả confirmatory."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
