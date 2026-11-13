"""Thăm dò (b-lite): thực thể cụm khái niệm (không LLM) có nâng trần kết nối bridge khái niệm mà không thêm nhiễu không?
Chỉ đọc corpus v2 + đồ thị MENTIONS; không API, không qa_test (cặp test chỉ dùng để kiểm tra không thoái lui ở bridge tên riêng, không chọn cấu hình).
Quy tắc quyết định KHAI BÁO TRƯỚC: coi là 'có triển vọng' nếu 1-bước trên cặp khái niệm >= 50% VÀ cặp ngẫu nhiên 1-bước <= 1%
VÀ bridge tên riêng DEV/test vẫn 100%. Chạy: python -X utf8 scripts/phrase_connectivity_audit.py"""
import sys, json, re, random
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.kg.phrases import paper_phrases, top_salient, entity_index
TRIAGE = [("2506.07159", "2403.02833"), ("2405.12983", "2303.00628"), ("2511.11592", "2310.05858"), ("2412.10251", "2412.09800"), ("2407.11370", "2306.00697"),
          ("2410.17338", "2305.18450"), ("2312.15204", "2309.06440"), ("2509.00528", "2305.13459"), ("2406.16751", "2310.13812"), ("2508.21353", "2404.19261")]


def load(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def concept_pairs():
    from openpyxl import load_workbook
    pid = lambda c: re.match(r"\[(.*?)\]", c).group(1); ok = set()
    for f in sorted((ROOT / "data/eval").glob("bridge_hard_verified*.xlsx")):
        for r in load_workbook(f)["Viết tay"].iter_rows(min_row=2, values_only=True):
            if r[16] == "OK": ok.add((pid(r[2]), pid(r[3])))
    return sorted(ok)


def main():
    manifest = json.loads((ROOT / "data/eval/benchmark_v2_assisted/manifest.json").read_text(encoding="utf-8"))
    papers = load(ROOT / manifest["corpus"].replace("\\", "/")); ids = [p["paper_id"] for p in papers]; print("papers", len(papers), flush=True)
    M = json.load(open(ROOT / "data/kg/reference_mentions_v2_assisted.json", encoding="utf-8"))
    mentions = {p: {e for e in v if len(M["entity_papers"][e]) >= 2} for p, v in M["paper_entities"].items()}
    allp = paper_phrases(papers)
    configs = {"mentions(baseline)": mentions, "phrase_all_df2-200": entity_index(allp, 2, 200),
               "phrase_top10": entity_index(top_salient(allp, 10), 2, 200), "phrase_top20": entity_index(top_salient(allp, 20), 2, 200)}
    configs["union_mentions+top10"] = {p: mentions.get(p, set()) | configs["phrase_top10"].get(p, set()) for p in ids}
    sets = {"concept_llm_verified": concept_pairs(), "concept_triage": TRIAGE}
    for nm, f in [("named_bridge_dev", "qa_dev"), ("named_bridge_test(no selection)", "qa_test")]:
        sets[nm] = [(x["hop1_id"], x["hop2_id"]) for x in load(ROOT / f"data/eval/benchmark_v2_assisted/{f}.jsonl") if x["type"] == "bridge"]
    random.seed(0); sets["random_pairs"] = [(random.choice(ids), random.choice(ids)) for _ in range(5000)]
    report = {}
    for cname, ent in configs.items():
        size = Counter(e for s in ent.values() for e in s); row = dict(n_entities=len(size), mean_entities_per_paper=round(sum(len(s) for s in ent.values()) / len(ent), 2),
                                                                     n_edges=sum(len(s) for s in ent.values()), pairs={})
        for sname, pairs in sets.items():
            conn = [(a, b) for a, b in pairs if ent.get(a, set()) & ent.get(b, set())]
            row["pairs"][sname] = dict(n=len(pairs), one_hop=len(conn), rate=round(len(conn) / max(len(pairs), 1), 4))
        if cname != "mentions(baseline)":
            row["examples_shared_concept_llm_verified"] = [dict(pair=[a, b], shared=sorted(ent[a] & ent[b])[:4]) for a, b in sets["concept_llm_verified"] if ent.get(a, set()) & ent.get(b, set())][:12]
        report[cname] = row
        print(f"\n{cname}: entities={row['n_entities']} edges={row['n_edges']} mean/paper={row['mean_entities_per_paper']}")
        for sname, v in row["pairs"].items(): print(f"   {sname:34s} {v['one_hop']}/{v['n']} ({v['rate']:.1%})")
    out = ROOT / "results/analysis/phrase_connectivity_audit.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in ["phrase_top10", "phrase_top20"]:
        print(f"\nVí dụ cụm chung ({c}) trên cặp khái niệm:")
        for e in report[c]["examples_shared_concept_llm_verified"]: print("  ", e["pair"], e["shared"])


if __name__ == "__main__": main()
