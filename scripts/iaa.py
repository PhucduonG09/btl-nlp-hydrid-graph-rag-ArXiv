"""Đồng thuận giữa hai annotator + danh sách bất đồng để thảo luận/phán quyết.
  python scripts/iaa.py [--dir data/eval/annotation]
Khớp mục A↔B: tên khớp lỏng HOẶC evidence trùng ≥50% token (một-một, tham lam).
Sau khi điền cột 'phán_quyết' trong results/analysis/iaa_disagreements.csv → chạy scripts/build_gold.py"""
import sys, json, csv, argparse, collections, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openpyxl import load_workbook
from src.kg.aliases import match
TAGS = ["GENERIC", "UNNAMED", "BASELINE", "STUDIED", "OTHER"]


def toks(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower()))
def ev_match(a, b):
    x, y = toks(a), toks(b)
    return bool(x and y) and len(x & y) / min(len(x), len(y)) >= 0.5


REL_FIX = {"EVALUATED_ON": "EVALUATES_ON", "EVALUATE_ON": "EVALUATES_ON", "PROPOSE": "PROPOSES", "USE": "USES", "ADDRESS": "ADDRESSES"}


def read(path):
    ws = load_workbook(path, data_only=True)["Gán nhãn"]
    out = collections.defaultdict(list)
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid, _, rel, tgt, ev, tag, note = (list(row) + [None] * 7)[:7]
        if pid and rel:
            rel = str(rel).strip().upper(); rel = REL_FIX.get(rel, rel)
            out[str(pid)].append({"relation": rel, "target": (tgt or "").strip(), "evidence": (ev or "").strip(),
                                  "tags": {t.strip().upper() for t in str(tag or "").replace(",", ";").split(";") if t.strip()},
                                  "note": str(note or "")})
            bad = out[str(pid)][-1]["tags"] - set(TAGS)
            if bad:      # tag lạ (thường do gõ nhầm cột): bỏ khỏi tag và cảnh báo
                out[str(pid)][-1]["tags"] -= bad
                print(f"CẢNH BÁO {Path(path).name} {pid}: tag không hợp lệ bị bỏ: {[t[:40] for t in bad]}", file=sys.stderr)
    return out


def kappa(x, y):
    n = len(x)
    if n == 0: return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in set(x) | set(y))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def fmt(x): return f"{x['relation']}:{x['target']} [{';'.join(sorted(x['tags']))}]"


def compare(A, B, ignore=frozenset()):
    """So hai danh sách mục của cùng một paper. Trả về pairs, onlyA, onlyB, none_diff, rows(bất đồng)."""
    nA = any(x["relation"] == "NONE" for x in A); nB = any(x["relation"] == "NONE" for x in B)
    Ai = [x for x in A if x["relation"] != "NONE"]; Bi = [x for x in B if x["relation"] != "NONE"]
    used, pairs = set(), []
    for i, x in enumerate(Ai):
        for j, y in enumerate(Bi):
            if j not in used and (match(x["target"], y["target"], True) or ev_match(x["evidence"], y["evidence"])):
                used.add(j); pairs.append((i, j)); break
    mi = {i for i, _ in pairs}
    onlyA = [i for i in range(len(Ai)) if i not in mi]; onlyB = [j for j in range(len(Bi)) if j not in used]
    nd = nA != nB
    KEEPABLE = {"BASELINE", "GENERIC", "STUDIED"}     # mục có tag lọc được → giữ không hại (nguyên tắc gán dễ)
    rows, sugg = [], []
    if nd:      # paper có NONE khác: chỉ một dòng bất đồng; không liệt kê từng mục của phía còn lại
        note = " ".join(x["note"] for x in (A if nA else B) if x["relation"] == "NONE").lower()
        survey = bool(re.search(r"survey|position|review|khảo sát|luận điểm|thảo luận|tổng quan", note))
        rows.append(["NONE khác", "NONE" if nA else "không NONE", "NONE" if nB else "không NONE"])
        sugg.append(("A" if nA else "B") if survey else ("B" if nA else "A"))   # empirical → giữ phía có mục; survey/position → NONE
    else:
        for i, j in pairs:
            x, y = Ai[i], Bi[j]
            if x["relation"] != y["relation"]: rows.append(["khác relation", fmt(x), fmt(y)]); sugg.append("")
            elif (x["tags"] - ignore) != (y["tags"] - ignore):
                tx, ty = x["tags"] - ignore, y["tags"] - ignore
                rows.append(["khác tag", fmt(x), fmt(y)]); sugg.append("A" if tx > ty else "B" if ty > tx else "")
        for i in onlyA: rows.append(["chỉ A có", fmt(Ai[i]), ""]); sugg.append("giữ" if Ai[i]["tags"] & KEEPABLE else "")
        for j in onlyB: rows.append(["chỉ B có", "", fmt(Bi[j])]); sugg.append("giữ" if Bi[j]["tags"] & KEEPABLE else "")
    return {"pairs": pairs, "onlyA": onlyA, "onlyB": onlyB, "noneA": nA, "noneB": nB, "none_diff": nd, "Ai": Ai, "Bi": Bi,
            "rows": rows, "sugg": sugg}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(ROOT / "data/eval/annotation"))
    ap.add_argument("--ignore-tags", default="", help="tag bỏ qua khi tính bất đồng, vd UNNAMED,OTHER")
    ap.add_argument("--dis-out", default=str(ROOT / "results/analysis/iaa_disagreements.csv")); a = ap.parse_args()
    IGN = frozenset(t for t in a.ignore_tags.split(",") if t)
    d = Path(a.dir); man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    RA, RB = read(d / "annotator_A.xlsx"), read(d / "annotator_B.xlsx")
    papers = [p for p in man["overlap"] if p in RA and p in RB]
    print(f"paper chung: {len(man['overlap'])}, cả hai đã gán: {len(papers)}")
    if not papers: sys.exit("Chưa có paper nào cả hai cùng gán.")
    nA = nB = nM = 0; relA, relB = [], []; tp = collections.defaultdict(lambda: ([], []))
    um, tot = collections.Counter(), collections.Counter(); dis = []; noneA, noneB = [], []
    for p in papers:
        c = compare(RA[p], RB[p], IGN); noneA.append(c["noneA"]); noneB.append(c["noneB"])
        Ai, Bi = c["Ai"], c["Bi"]; nA += len(Ai); nB += len(Bi); nM += len(c["pairs"])
        for i, j in c["pairs"]:
            x, y = Ai[i], Bi[j]; relA.append(x["relation"]); relB.append(y["relation"])
            for t in TAGS: tp[t][0].append(t in x["tags"]); tp[t][1].append(t in y["tags"])
        for i, x in enumerate(Ai):
            for t in (x["tags"] or {"(không tag)"}): tot[t] += 1; um[t] += i in c["onlyA"]
        for j, y in enumerate(Bi):
            for t in (y["tags"] or {"(không tag)"}): tot[t] += 1; um[t] += j in c["onlyB"]
        dis += [[p] + r + [sg] for r, sg in zip(c["rows"], c["sugg"])]
    f1 = 2 * nM / (nA + nB) if nA + nB else 0
    print(f"\nMục: A={nA}, B={nB}, khớp={nM}  →  F1 đồng thuận (entity-level) = {f1:.3f}   [mục tiêu ≥ 0.8]")
    print(f"Trên cặp khớp: relation giống nhau {sum(a == b for a, b in zip(relA, relB))}/{len(relA)}, κ(relation) = {kappa(relA, relB):.2f}")
    print(f"NONE (paper không có quan hệ): κ = {kappa(noneA, noneB):.2f}  (A: {sum(noneA)}, B: {sum(noneB)} paper)")
    print("\nTag — κ trên cặp khớp | mục mang tag mà chỉ một người đưa vào (bất đồng về việc CÓ gán hay không):")
    for t, (x, y) in tp.items():
        print(f"  {t:9s} κ={kappa(x, y):5.2f}  (A={sum(x)}, B={sum(y)}) | mục có tag, không khớp: {um[t]}/{tot[t]}")
    print(f"  (không tag) mục không khớp: {um['(không tag)']}/{tot['(không tag)']}")
    print("\nCách đọc: κ ≥ 0.6 → tag/quy tắc áp dụng được nhất quán; κ < 0.4 → quy tắc mơ hồ, viết lại trước khi dùng.")
    out = Path(a.dis_out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["paper_id", "loại", "A", "B", "gợi_ý", "phán_quyết"]); w.writerows([r + [""] for r in dis])
    print(f"Ghi {len(dis)} bất đồng -> {out}\nCột gợi_ý theo nguyên tắc gán dễ (build_gold.py --accept-suggestions để dùng). Điền cột phán_quyết cho dòng còn trống: 'khác relation/khác tag/NONE khác' → A | B | bỏ ;  'chỉ A có/chỉ B có' → giữ | bỏ")


if __name__ == "__main__":
    main()
