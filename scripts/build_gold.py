"""Ghép gold cuối từ hai file annotator + cột phán_quyết.   python scripts/build_gold.py
Đầu ra: data/eval/gold_v4.csv (paper_id,relation,target,evidence,tag). Dừng và liệt kê nếu còn bất đồng chưa phán quyết."""
import sys, csv, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from iaa import read, compare, fmt
from rule_impact import is_named


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default=str(ROOT / "data/eval/annotation"))
    ap.add_argument("--dis", default=str(ROOT / "results/analysis/iaa_disagreements.csv"))
    ap.add_argument("--ignore-tags", default=""); ap.add_argument("--accept-suggestions", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data/eval/gold_v4.csv")); a = ap.parse_args()
    IGN = frozenset(t for t in a.ignore_tags.split(",") if t)
    d = Path(a.dir); man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    RA, RB = read(d / "annotator_A.xlsx"), read(d / "annotator_B.xlsx")
    dec = {(r["paper_id"], r["loại"], r["A"], r["B"]): (r["phán_quyết"].strip() or (r.get("gợi_ý", "").strip() if a.accept_suggestions else ""))
           for r in csv.DictReader(open(a.dis, encoding="utf-8-sig"))}
    rows, missing = [], []

    def emit(p, x):
        tags = set(x["tags"]) - IGN
        if "UNNAMED" in IGN and x["relation"] in ("PROPOSES", "USES", "EVALUATES_ON") and not is_named(x["target"]):
            tags.add("UNNAMED")          # tính tự động từ tên thay vì annotator tự phán
        rows.append([p, x["relation"], x["target"], x["evidence"], ";".join(sorted(tags))])
    for p in man["only_A"]:
        for x in RA.get(p, []): emit(p, x)
    for p in man["only_B"]:
        for x in RB.get(p, []): emit(p, x)
    for p in man["overlap"]:
        if p not in RA or p not in RB: missing.append(f"{p}: chưa đủ cả hai người gán"); continue
        c = compare(RA[p], RB[p], IGN); Ai, Bi = c["Ai"], c["Bi"]
        if c["none_diff"]:
            v = dec.get((p, "NONE khác", "NONE" if c["noneA"] else "không NONE", "NONE" if c["noneB"] else "không NONE"), "")
            if v not in ("A", "B"): missing.append(f"{p}: NONE khác"); continue
            for x in (RA[p] if v == "A" else RB[p]): emit(p, x)       # lấy toàn bộ phía được chọn (NONE hoặc các mục)
            continue
        elif c["noneA"]: emit(p, next(x for x in RA[p] if x["relation"] == "NONE"))
        for i, j in c["pairs"]:
            x, y = Ai[i], Bi[j]
            if x["relation"] == y["relation"] and (x["tags"] - IGN) == (y["tags"] - IGN): emit(p, {**x, "tags": x["tags"] | y["tags"]}); continue
            key = (p, "khác relation" if x["relation"] != y["relation"] else "khác tag", fmt(x), fmt(y))
            v = dec.get(key, "")
            if v == "A": emit(p, x)
            elif v == "B": emit(p, y)
            elif v != "bỏ": missing.append(f"{p}: {key[1]} {key[2]} | {key[3]}")
        for i in c["onlyA"]:
            v = dec.get((p, "chỉ A có", fmt(Ai[i]), ""), "")
            if v == "giữ": emit(p, Ai[i])
            elif v != "bỏ": missing.append(f"{p}: chỉ A có {fmt(Ai[i])}")
        for j in c["onlyB"]:
            v = dec.get((p, "chỉ B có", "", fmt(Bi[j])), "")
            if v == "giữ": emit(p, Bi[j])
            elif v != "bỏ": missing.append(f"{p}: chỉ B có {fmt(Bi[j])}")
    if missing:
        print(f"CHƯA XONG: {len(missing)} mục chưa có phán quyết hợp lệ, ví dụ:"); print("\n".join(missing[:15])); sys.exit(1)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["paper_id", "relation", "target", "evidence", "tag"]); w.writerows(rows)
    print(f"Ghi {len(rows)} dòng, {len({r[0] for r in rows})} paper -> {a.out}")


if __name__ == "__main__":
    main()
