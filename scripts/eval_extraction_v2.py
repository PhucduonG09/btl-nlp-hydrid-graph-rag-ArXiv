"""Chấm extractor v2: theo từng relation, hai mức khớp, hỗ trợ bật/tắt quy tắc qua tag, CI bootstrap theo paper.
  # gold cũ (idx):
  python scripts/eval_extraction_v2.py --pred data/kg/dev_extract_out_v3.jsonl --gold data/eval/gold_dev20.csv --papers data/eval/dev_extract.jsonl
  # gold mới (build_gold.py), bật quy tắc "bỏ GENERIC và BASELINE":
  python scripts/eval_extraction_v2.py --pred data/kg/new_out.jsonl --gold data/eval/gold_v4.csv --exclude-tags GENERIC,BASELINE
Khớp 'entity' = tên khớp lỏng HOẶC evidence trùng ≥50% token (bỏ qua relation); 'typed' = entity + đúng relation.
--exclude-tags: gold mang tag đó bị bỏ khỏi gold, và dự đoán khớp với chúng KHÔNG tính TP/FP (ignore) → so sánh công bằng khi bật/tắt quy tắc."""
import sys, csv, json, re, argparse, random
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kg.aliases import match
RELS4 = ["PROPOSES", "USES", "EVALUATES_ON", "ADDRESSES"]


def toks(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))
def same_entity(p, g):
    x, y = toks(p.get("evidence", "")), toks(g.get("evidence", ""))
    return match(p["target"], g["target"], True) or (bool(x and y) and len(x & y) / min(len(x), len(y)) >= 0.5)


def make_matcher(mode):
    """principled: mục gold CÓ tên → bắt buộc khớp tên; mục gold UNNAMED → tên HOẶC evidence Jaccard>=0.7.
    name: chỉ tên (chặn dưới). lenient: tên HOẶC evidence >=50% của câu ngắn (chặn trên, có thể ghép nhầm 2 thực thể cùng câu)."""
    def jac(p, g):
        x, y = toks(p.get("evidence", "")), toks(g.get("evidence", ""))
        return bool(x and y) and len(x & y) / len(x | y) >= 0.7
    if mode == "name": return lambda p, g: match(p["target"], g["target"], True)
    if mode == "lenient": return same_entity
    return lambda p, g: match(p["target"], g["target"], True) or ("UNNAMED" in g.get("tags", ()) and jac(p, g))


def load_gold(path, papers_file):
    rd = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    gold = defaultdict(list)
    if rd and "idx" in rd[0]:                                   # định dạng cũ
        assert papers_file, "gold cũ (idx) cần --papers (dev_extract.jsonl / test_extract.jsonl)"
        ps = [json.loads(l)["paper_id"] for l in open(papers_file, encoding="utf-8") if l.strip()]
        gold = {p: [] for p in ps}
        for r in rd: gold[ps[int(r["idx"]) - 1]].append({**r, "tags": set()})
        return gold, "old"
    for r in rd:
        gold[r["paper_id"]]
        if r["relation"] != "NONE":
            gold[r["paper_id"]].append({**r, "tags": {t for t in (r.get("tag") or "").split(";") if t}})
    return dict(gold), "new"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--gold", required=True); ap.add_argument("--papers")
    ap.add_argument("--rels", default="auto"); ap.add_argument("--exclude-tags", default="")
    ap.add_argument("--subset", default="", help="overlap | single (chỉ với gold_v4: 30 paper đã đối chiếu / 30 paper một người gán)")
    ap.add_argument("--manifest", default=str(ROOT / "data/eval/annotation/manifest.json"))
    ap.add_argument("--match", default="principled", choices=["principled", "name", "lenient"])
    ap.add_argument("--label", default="run"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    se = make_matcher(a.match)
    gold, fmt = load_gold(a.gold, a.papers)
    if a.subset:
        man = json.load(open(a.manifest, encoding="utf-8"))
        keep = set(man["overlap"]) if a.subset == "overlap" else set(man["only_A"]) | set(man["only_B"])
        gold = {k: v for k, v in gold.items() if k in keep}
    rels = (["PROPOSES", "USES"] if fmt == "old" else RELS4) if a.rels == "auto" else a.rels.split(",")
    excl = {t for t in a.exclude_tags.split(",") if t}
    pred = {}
    for l in open(a.pred, encoding="utf-8"):
        if l.strip(): d = json.loads(l); pred[d["paper_id"]] = d.get("relations", [])
    miss = [p for p in gold if p not in pred]
    if miss: print(f"CẢNH BÁO: thiếu dự đoán cho {len(miss)} paper")

    # đếm theo (paper, relation): tp_p, n_p (precision) và tp_g, n_g (recall) cho 2 mức khớp
    cnt = {m: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])) for m in ("entity", "typed")}
    fp_empty = 0
    for pid, g_all in gold.items():
        g = [x for x in g_all if x["relation"] in rels and not (x["tags"] & excl)]
        ign = [x for x in g_all if x["relation"] in rels and (x["tags"] & excl)]
        ps = [x for x in pred.get(pid, []) if x["relation"] in rels]
        for x in ps:
            if any(se(x, y) for y in ign) and not any(se(x, y) for y in g): continue   # ignore
            for m in cnt:
                ok = any(se(x, y) and (m == "entity" or x["relation"] == y["relation"]) for y in g)
                c = cnt[m][pid][x["relation"]]; c[0] += ok; c[1] += 1
            if not g_all: fp_empty += 1
        for y in g:
            for m in cnt:
                ok = any(se(x, y) and (m == "entity" or x["relation"] == y["relation"]) for x in ps)
                c = cnt[m][pid][y["relation"]]; c[2] += ok; c[3] += 1

    def prf(rows):
        tp_p = sum(r[0] for r in rows); n_p = sum(r[1] for r in rows); tp_g = sum(r[2] for r in rows); n_g = sum(r[3] for r in rows)
        P = tp_p / n_p if n_p else float("nan"); R = tp_g / n_g if n_g else float("nan")
        F = 2 * P * R / (P + R) if P and R and P + R else 0.0
        return P, R, F, n_p, n_g

    pids = list(gold); rng = random.Random(0)
    print(f"[{a.label}] {len(pids)} paper, gold={sum(len([x for x in v if x['relation'] in rels and not (x['tags'] & excl)]) for v in gold.values())} mục, "
          f"khớp: {a.match}, relation chấm: {','.join(rels)}" + (f", bỏ tag: {','.join(sorted(excl))}" if excl else ""))
    result = {"label": a.label, "match": a.match, "pred": a.pred, "gold": a.gold, "exclude_tags": sorted(excl)}
    for m in ("entity", "typed"):
        print(f"\n== mức khớp: {m} ==\n{'relation':13s}{'P':>7}{'R':>7}{'F1':>7}{'n_pred':>8}{'n_gold':>8}")
        for rel in ["ALL"] + rels:
            sel = lambda pp: [cnt[m][p][r] for p in pp for r in (rels if rel == "ALL" else [rel]) if r in cnt[m][p]]
            P, R, F, npd, ng = prf(sel(pids))
            ci = ""
            if rel == "ALL" and a.boot:
                fs = sorted(prf(sel([rng.choice(pids) for _ in pids]))[2] for _ in range(a.boot))
                ci = f"   F1 95% CI [{fs[int(.025*a.boot)]:.2f}, {fs[int(.975*a.boot)]:.2f}]"
            print(f"{rel:13s}{P:7.3f}{R:7.3f}{F:7.3f}{npd:8d}{ng:8d}{ci}")
            result[f"{m}/{rel}"] = {"P": round(P, 4), "R": round(R, 4), "F1": round(F, 4), "n_pred": npd, "n_gold": ng}
    n_empty = sum(1 for v in gold.values() if not v)
    print(f"\npaper không có gold: {n_empty}; dự đoán rơi vào đó (tính FP): {fp_empty}")
    out = ROOT / "results/extraction_eval_v2.jsonl"
    with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
