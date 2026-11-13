"""Routing theo truy vấn (offline, không BGE, không API, không test): dùng graph chỉ khi câu hỏi có thực thể liên kết được (df<=10, đúng điều kiện linked_seeds của engine);
ngược lại dùng RRF. Quy tắc suy ra từ logic engine, KHÔNG tune ngưỡng. Đọc xếp hạng đã lưu của DEV (dev_01/retrieval.jsonl).
Chạy: python -X utf8 scripts/routing_dev.py"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.kg.mentions import MentionIndex
from src.eval.benchmark_metrics import summarize
load = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
B = ROOT / "results/benchmark_v2_assisted/dev_01"
qa = load(ROOT / "data/eval/benchmark_v2_assisted/qa_dev.jsonl"); saved = {r["id"]: r for r in load(B / "retrieval.jsonl")}
cfg = json.loads((B / "selected_config.json").read_text(encoding="utf-8")); rk = lambda q, name: saved[q["id"]]["rankings"][name]
M = MentionIndex(json.load(open(ROOT / "data/kg/reference_mentions_v2_assisted.json", encoding="utf-8")))
linked = {}
for q in qa:
    ents = [e for e in M.query_entities(q["question"]) if len(M.entity_papers[e]) <= 10]; linked[q["id"]] = bool(ents)
rr = {q["id"]: rk(q, cfg["rrf"]) for q in qa}; gr = {q["id"]: rk(q, cfg["graph"]) for q in qa}
route = {q["id"]: (gr[q["id"]] if linked[q["id"]] else rr[q["id"]]) for q in qa}
res = {n: summarize(qa, r) for n, r in [("rrf", rr), ("graph", gr), ("route(linked->graph else rrf)", route)]}
cnt = {t: (sum(linked[q["id"]] for q in qa if q["type"] == t), sum(q["type"] == t for q in qa)) for t in ("single", "bridge")}
print("câu có thực thể liên kết (graph được bật):", cnt)
for n, s in res.items():
    print(n, {g: {k: round(v, 3) for k, v in s[g].items() if k in ("R@1", "MRR@10", "R@10", "All@10")} for g in ("single", "bridge")})
