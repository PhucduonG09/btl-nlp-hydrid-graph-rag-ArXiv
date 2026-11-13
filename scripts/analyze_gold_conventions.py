"""Tách nguyên nhân F1 thấp của extractor: lệch quy ước nhãn quan hệ, lệch cách đặt tên, hay lỗi thật.
  python scripts/analyze_gold_conventions.py
Các cách chấm (cùng pred, cùng gold):
  A  hiện tại : đúng relation + tên khớp lỏng, chỉ PROPOSES/USES
  B  bỏ nhãn  : chỉ cần tên khớp lỏng, gộp PROPOSES/USES/EVALUATES_ON
  C  theo evidence: đúng relation + câu evidence trùng ≥50% token (không phụ thuộc cách đặt tên)
  D  B hoặc C : tên HOẶC evidence khớp, không cần đúng relation"""
import sys, csv, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kg.aliases import match

SETS = {"dev": ("data/eval/dev_extract.jsonl", "data/eval/gold_dev20.csv", "data/kg/dev_extract_out_v3.jsonl"),
        "test": ("data/eval/test_extract.jsonl", "data/eval/gold_test.csv", "data/kg/test_extract_out_v3.jsonl")}
KEEP = {"PROPOSES", "USES", "EVALUATES_ON"}


def toks(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))
def ev_match(a, b):
    x, y = toks(a), toks(b)
    return bool(x and y) and len(x & y) / min(len(x), len(y)) >= 0.5


def load(name):
    devf, goldf, predf = [ROOT / p for p in SETS[name]]
    dev = [json.loads(l) for l in open(devf, encoding="utf-8") if l.strip()]
    pred = {}
    for l in open(predf, encoding="utf-8"):
        if l.strip():
            d = json.loads(l); pred[d["paper_id"]] = d.get("relations", [])
    gold = {p["paper_id"]: [] for p in dev}
    for r in csv.DictReader(open(goldf, encoding="utf-8-sig")):
        gold[dev[int(r["idx"]) - 1]["paper_id"]].append(r)
    return dev, pred, gold


def prf(tp_p, n_p, tp_g, n_g):
    P = tp_p / n_p if n_p else 0; R = tp_g / n_g if n_g else 0
    return P, R, (2 * P * R / (P + R) if P + R else 0)


def score(pred, gold, mode):
    tpp = npp = tpg = ng = 0
    for pid, g in gold.items():
        rels = {"A": {"PROPOSES", "USES"}}.get(mode, KEEP)
        gl = [x for x in g if x["relation"] in {"PROPOSES", "USES"}]
        pl = [x for x in pred.get(pid, []) if x["relation"] in rels]
        def same(p, x):
            nm = match(p["target"], x["target"], True)
            ev = ev_match(p.get("evidence", ""), x.get("evidence", ""))
            rel = p["relation"] == x["relation"]
            return {"A": rel and nm, "B": nm, "C": rel and ev, "D": nm or ev}[mode]
        npp += len(pl); ng += len(gl)
        tpp += sum(any(same(p, x) for x in gl) for p in pl)
        tpg += sum(any(same(p, x) for p in pl) for x in gl)
    return prf(tpp, npp, tpg, ng) + (npp, ng)


for name in SETS:
    dev, pred, gold = load(name)
    print(f"\n=== {name}: {len(dev)} paper, {sum(len(v) for v in gold.values())} gold ===")
    for m, desc in [("A", "hiện tại (relation + tên)"), ("B", "bỏ nhãn quan hệ, chỉ tên"),
                    ("C", "relation + evidence"), ("D", "tên HOẶC evidence")]:
        P, R, F, npd, ng = score(pred, gold, m)
        print(f"  {m} {desc:28s} P={P:.3f} R={R:.3f} F1={F:.3f}  (n_pred={npd}, n_gold={ng})")
    empty = [p["paper_id"] for p in dev if not gold[p["paper_id"]]]
    fp_empty = sum(1 for pid in empty for r in pred.get(pid, []) if r["relation"] in {"PROPOSES", "USES"})
    print(f"  paper không có gold: {len(empty)}; số dự đoán PROPOSES/USES rơi vào đó (mặc định bị tính FP): {fp_empty}")
    # thiếu theo relation + dataset-nhãn-USES trong gold
    ds = [(g["target"], pid) for pid, v in gold.items() for g in v if g["relation"] == "USES"
          and any(match(g["target"], r["target"], True) and r["relation"] == "EVALUATES_ON" for r in pred.get(pid, []))]
    print(f"  gold nhãn USES nhưng extractor gán EVALUATES_ON (lệch quy ước): {len(ds)} → {[t for t, _ in ds]}")
    conf = {}
    for pid, v in gold.items():
        for g in v:
            for r in pred.get(pid, []):
                if match(g["target"], r["target"], True) and r["relation"] != g["relation"]:
                    conf[(g["relation"], r["relation"])] = conf.get((g["relation"], r["relation"]), 0) + 1
    print("  gold→pred đúng tên nhưng khác relation:", conf)
