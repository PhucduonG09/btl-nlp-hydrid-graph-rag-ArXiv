"""Mô phỏng tác động của từng quy tắc gán nhãn lên KG (500 paper pilot, dự đoán v3) trước khi bạn chốt.
  python scripts/rule_impact.py [--extract data/kg/pilot_extract_v3.jsonl]
Với mỗi quy tắc: bỏ các cạnh bị quy tắc loại, so sánh KG trước/sau.
Chỉ số chính: 'cạnh bị loại mà có tính cầu nối' = cạnh có concept xuất hiện ở ≥2 paper.
  thấp  → quy tắc rẻ (chủ yếu dọn nhiễu)   cao → quy tắc đắt (mất kết nối, có thể quá chặt).
Heuristic 'có tên'/'nền' chỉ là xấp xỉ: hãy xem file mẫu results/analysis/rule_impact_samples.csv để tự kiểm tra bằng mắt."""
import sys, re, json, csv, random, argparse, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.kg.aliases import canon, GENERIC

REL4 = {"PROPOSES", "USES", "EVALUATES_ON", "ADDRESSES"}
BG = re.compile(r"\b(existing|prior|previous|recent(ly)?|state-of-the-art|baseline|compared? (to|with)|such as|e\.g\.|like|"
                r"typically|commonly|widely|popular|related work|overlooking|traditional)\b", re.I)


def is_named(t):
    """Có tên riêng: có chữ hoa ở giữa/viết tắt, chữ số, hoặc 1 từ viết hoa chữ đầu (Wikipedia)."""
    toks = t.split()
    if any(re.search(r"[A-Z]{2,}|[a-z][A-Z]|\d", w) for w in toks): return True
    return len(toks) == 1 and t[:1].isupper()


RULES = {
    "R3a USES chung chung (LLM, CNN-loại GENERIC...) → bỏ":
        lambda r: r["relation"] == "USES" and canon(r["target"]) in GENERIC,
    "R3b USES không có tên riêng (xấp xỉ) → bỏ":
        lambda r: r["relation"] == "USES" and not is_named(r["target"]),
    "R4  PROPOSES không có tên (cụm mô tả) → bỏ":
        lambda r: r["relation"] == "PROPOSES" and not is_named(r["target"]) and len(r["target"].split()) >= 3,
    "R6  USES có dấu hiệu nền/baseline trong evidence → bỏ":
        lambda r: r["relation"] == "USES" and bool(BG.search(r.get("evidence", ""))),
    "R1  gộp EVALUATES_ON vào USES (chỉ đổi tên nhãn)":
        None,
}


class UF:
    def __init__(s): s.p = {}
    def f(s, x):
        s.p.setdefault(x, x)
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def kg_stats(edges, dmax=20):
    df = collections.defaultdict(set)
    for pid, c, rel in edges: df[c].add(pid)
    useful = {c for c, ps in df.items() if 2 <= len(ps) <= dmax}
    uf = UF(); pairs = set()
    for c in useful:
        ps = sorted(df[c])
        for p in ps[1:]: uf.u(ps[0], p)
        pairs.update((a, b) for i, a in enumerate(ps) for b in ps[i + 1:])
    comp = collections.Counter(uf.f(p) for p in uf.p)
    return {"edges": len(edges), "concepts": len(df), "shared(≥2)": sum(len(p) >= 2 for p in df.values()),
            "useful_shared(2-20)": len(useful), "bridge_pairs": len(pairs),
            "papers_in_largest_comp": max(comp.values()) if comp else 0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", default=str(ROOT / "data/kg/pilot_extract_v3.jsonl"))
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.extract, encoding="utf-8") if l.strip()]
    ins = [r["usage"]["in"] for r in rows if r.get("usage")]; outs = [r["usage"]["out"] for r in rows if r.get("usage")]
    items = [dict(r, pid=d["paper_id"], c=canon(r["target"])) for d in rows for r in d["relations"] if r["relation"] in REL4]
    df = collections.defaultdict(set)
    for it in items: df[it["c"]].add(it["pid"])
    base = kg_stats([(i["pid"], i["c"], i["relation"]) for i in items])
    print(f"Token thực tế/paper khi trích xuất: in≈{sum(ins)/len(ins):.0f}, out≈{sum(outs)/len(outs):.0f} (n={len(ins)})")
    print("\nKG gốc:", base)
    samples = []; random.seed(0)
    for name, fn in RULES.items():
        if fn is None:
            ds = [i for i in items if i["relation"] == "EVALUATES_ON"]
            sh = sum(len(df[i["c"]]) >= 2 for i in ds)
            print(f"\n[{name}] {len(ds)} cạnh EVALUATES_ON ({sh} cạnh dataset dùng chung ≥2 paper). Kết nối KG không đổi; "
                  f"chỉ khác cách đặt câu hỏi/cấu trúc truy vấn.")
            continue
        rem = [i for i in items if fn(i)]
        keep = [(i["pid"], i["c"], i["relation"]) for i in items if not fn(i)]
        st = kg_stats(keep)
        bridged = sum(len(df[i["c"]]) >= 2 for i in rem)
        n_rel = sum(1 for i in items if i["relation"] == rem[0]["relation"]) if rem else 1
        print(f"\n[{name}]\n  loại {len(rem)} cạnh ({100*len(rem)/n_rel:.0f}% số cạnh {rem[0]['relation'] if rem else ''}); "
              f"trong đó có tính cầu nối (concept ≥2 paper): {bridged} ({100*bridged/max(1,len(rem)):.0f}%)")
        print(f"  KG sau: shared {base['shared(≥2)']}→{st['shared(≥2)']}, useful_shared {base['useful_shared(2-20)']}→{st['useful_shared(2-20)']}, "
              f"bridge_pairs {base['bridge_pairs']}→{st['bridge_pairs']}, largest_comp {base['papers_in_largest_comp']}→{st['papers_in_largest_comp']}")
        for i in random.sample(rem, min(12, len(rem))):
            samples.append({"rule": name.split()[0], "paper": i["pid"], "relation": i["relation"], "target": i["target"],
                            "papers_sharing": len(df[i["c"]]), "evidence": i.get("evidence", "")[:140]})
    out = ROOT / "results/analysis/rule_impact_samples.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0])); w.writeheader(); w.writerows(samples)
    print("\nMẫu để tự kiểm tra bằng mắt ->", out)


if __name__ == "__main__":
    main()
