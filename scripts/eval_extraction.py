"""Chấm P/R/F1 của extractor so với gold_dev20.csv (dùng alias và khớp viết tắt).
  python scripts/eval_extraction.py --pred data/kg/dev_extract_out_v3.jsonl --label v3"""
import sys, csv, json, argparse
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kg.aliases import match, canon, GENERIC

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--dev", default=str(ROOT / "data/eval/dev_extract.jsonl"))
ap.add_argument("--gold", default=str(ROOT / "data/eval/gold_dev20.csv"))
ap.add_argument("--rels", default="PROPOSES,USES")
ap.add_argument("--drop-generic", action="store_true", help="bỏ USES mang khái niệm quá chung (LLM, deep learning...)")
ap.add_argument("--label", default="run")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()
RELS = set(a.rels.split(","))

dev = [json.loads(l) for l in open(a.dev, encoding="utf-8") if l.strip()]
pred = {}
for l in open(a.pred, encoding="utf-8"):
    if l.strip():
        d = json.loads(l); pred[d["paper_id"]] = d.get("relations", [])
rows = defaultdict(list)
for r in csv.DictReader(open(a.gold, encoding="utf-8-sig")):
    rows[int(r["idx"])].append(r)
gold = {p["paper_id"]: rows.get(i, []) for i, p in enumerate(dev, 1)}


def items(rels, is_pred):
    out = []
    for r in rels:
        if r["relation"] not in RELS:
            continue
        if is_pred and a.drop_generic and r["relation"] == "USES" and canon(r["target"]) in GENERIC:
            continue
        out.append((r["relation"], r["target"]))
    return out


def prf(loose):
    same = lambda x, y: x[0] == y[0] and match(x[1], y[1], loose)
    tpp = tpg = npp = ng = 0
    for pid, g in gold.items():
        g, p = items(g, False), items(pred.get(pid, []), True)
        ng += len(g); npp += len(p)
        tpp += sum(any(same(x, y) for y in g) for x in p)
        tpg += sum(any(same(x, y) for y in p) for x in g)
    P = tpp / npp if npp else 0.0
    R = tpg / ng if ng else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"P": round(P, 3), "R": round(R, 3), "F1": round(F, 3), "n_pred": npp, "n_gold": ng}


res = {"label": a.label, "pred_file": a.pred, "drop_generic": a.drop_generic,
       "strict": prf(False), "loose": prf(True),
       "missing_papers": [p["paper_id"] for p in dev if p["paper_id"] not in pred]}
print(a.label, "| khớp chặt:", res["strict"], "\n", " " * len(a.label), "| khớp lỏng:", res["loose"])
if res["missing_papers"]:
    print("CẢNH BÁO: thiếu dự đoán cho", len(res["missing_papers"]), "paper")
if not a.quiet:
    for i, p in enumerate(dev, 1):
        g, q = items(gold[p["paper_id"]], False), items(pred.get(p["paper_id"], []), True)
        miss = [x for x in g if not any(x[0] == y[0] and match(x[1], y[1], True) for y in q)]
        extra = [y for y in q if not any(x[0] == y[0] and match(x[1], y[1], True) for x in g)]
        if miss or extra:
            print(f"[{i}] {p['paper_id']} | thiếu: {[(r, t) for r, t in miss]} | thừa: {[(r, t) for r, t in extra]}")
out = ROOT / "results" / "extraction_eval.jsonl"
out.parent.mkdir(exist_ok=True)
with open(out, "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
