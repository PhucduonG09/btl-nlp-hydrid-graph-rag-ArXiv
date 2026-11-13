"""Trần kết nối của graph MENTIONS: với mỗi cặp (A,B) 'A dựa vào B', graph có nối được hai paper không?
Không dùng LLM/embedding/QA. Chỉ đọc đồ thị MENTIONS (data/kg/reference_mentions_v2_assisted.json).
  python scripts/graph_connectivity_audit.py
Nhóm: benchmark v2 bridge (DEV/test), cặp 'concept' đã qua verify (LLM reviewer, CHƯA duyệt người), 10 cặp do trợ lý triage, đối chứng ngẫu nhiên.
Đầu ra: results/analysis/graph_connectivity_audit.json"""
import sys, json, re, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TRIAGE = [("2506.07159", "2403.02833"), ("2405.12983", "2303.00628"), ("2511.11592", "2310.05858"), ("2412.10251", "2412.09800"), ("2407.11370", "2306.00697"),
          ("2410.17338", "2305.18450"), ("2312.15204", "2309.06440"), ("2509.00528", "2305.13459"), ("2406.16751", "2310.13812"), ("2508.21353", "2404.19261")]


def main():
    M = json.load(open(ROOT / "data/kg/reference_mentions_v2_assisted.json", encoding="utf-8"))
    pe = {p: set(v) for p, v in M["paper_entities"].items()}; ep = M["entity_papers"]
    shared = lambda a, b: {e for e in pe.get(a, set()) & pe.get(b, set()) if len(ep[e]) >= 2}

    def two_hop(a, b):
        for e in pe.get(a, ()):
            if 2 <= len(ep[e]) <= 200:
                for m in ep[e]:
                    if m != a and pe.get(m, set()) & pe.get(b, set()): return True
        return False

    def stat(pairs):
        n = len(pairs); one = sum(bool(shared(a, b)) for a, b in pairs); two = sum(bool(shared(a, b)) or two_hop(a, b) for a, b in pairs)
        return dict(n=n, one_hop=one, one_hop_rate=round(one / max(n, 1), 3), up_to_two_hops=two, two_hop_rate=round(two / max(n, 1), 3))
    out = {}
    for nm, f in [("benchmark_v2_bridge_dev", "qa_dev"), ("benchmark_v2_bridge_test", "qa_test")]:
        q = [json.loads(l) for l in open(ROOT / f"data/eval/benchmark_v2_assisted/{f}.jsonl", encoding="utf-8")]
        out[nm] = stat([(x["hop1_id"], x["hop2_id"]) for x in q if x["type"] == "bridge"])
    pid = lambda c: re.match(r"\[(.*?)\]", c).group(1)
    ok = set()
    try:
        from openpyxl import load_workbook
        for f in sorted((ROOT / "data/eval").glob("bridge_hard_verified*.xlsx")):
            for r in load_workbook(f)["Viết tay"].iter_rows(min_row=2, values_only=True):
                if r[16] == "OK": ok.add((pid(r[2]), pid(r[3])))
    except ImportError: pass
    out["concept_verified_by_llm_reviewer"] = stat(sorted(ok)); out["concept_assistant_triage_kept"] = stat(TRIAGE)
    ids = list(pe); random.seed(0); out["control_random_pairs"] = stat([(random.choice(ids), random.choice(ids)) for _ in range(2000)])
    out["notes"] = ("Cặp concept: reviewer LLM, chưa duyệt người; triage do trợ lý, chưa duyệt người. n nhỏ. Hai hop dùng thực thể df 2..200; "
                    "đối chứng ngẫu nhiên 2 hop cao hơn mong đợi vì thực thể phổ biến. Đây là TRẦN: nối được trong đồ thị không bảo đảm truy xuất đúng.")
    (ROOT / "results/analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/analysis/graph_connectivity_audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict): print(f"{k:38} n={v['n']:4} 1-bước {v['one_hop']}/{v['n']} ({v['one_hop_rate']:.0%})  ≤2-bước {v['up_to_two_hops']}/{v['n']} ({v['two_hop_rate']:.0%})")


if __name__ == "__main__":
    main()
