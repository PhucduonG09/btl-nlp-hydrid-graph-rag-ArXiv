"""Kiểm tra KHÔNG GÂY HẠI + ablation khi thêm kênh CITES (Semantic Scholar) vào truy xuất lai. Không API, không qa_test.
Với mỗi câu: seed = top-3 RRF; ứng viên CITES = tham chiếu trực tiếp + láng giềng 2 bước (khai báo trước, w=[1,1]) hoặc chỉ 2 bước (thăm dò, thêm sau khi thấy kết quả mức cặp).
CHỐNG VÒNG TRÒN: với câu bridge, MỌI cạnh giữa các paper gold bị giấu khi dựng ứng viên (không để graph nhìn thấy cạnh A-B).
Hệ thống: rrf | graph (MENTIONS, đã khóa) | rrf+cites | graph+cites. Bộ: DEV v2 (122) và 7 câu bridge khó (hard7).
Tiêu chí không-gây-hại (khai báo trước): graph+cites không kém graph quá 3 điểm ở single MRR@10 và bridge All@10 trên DEV.
Chạy: python -X utf8 scripts/eval_cites_fusion.py --run-id cites_01"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.citation_cites_leaveout import cites_candidates
from scripts.eval_two_hop_baseline import SETS, load, rank_of, summarize, mcnemar
from src.retrieval.graph import fuse_rrf


class MaskedAdj:
    """Đồ thị trích dẫn vô hướng, ẩn các cạnh giữa tập paper 'mask' (gold của câu hiện tại)."""
    def __init__(self, adj, mask): self.adj, self.mask = adj, set(mask)
    def __getitem__(self, p):
        return {q for q in self.adj[p] if not (p in self.mask and q in self.mask)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); a = ap.parse_args()
    out = ROOT / "results/cites_fusion" / a.run_id
    if out.exists(): sys.exit("run-id đã tồn tại.")
    out.mkdir(parents=True)
    edges = sorted({tuple(e) for l in (ROOT / "data/external/citations_s2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() for e in json.loads(l)["edges"]})
    adj = defaultdict(set)
    for x, y in edges: adj[x].add(y); adj[y].add(x)
    from src.retrieval.engine import load_engine
    engine = load_engine(); systems = ["rrf", "graph", "rrf+cites", "graph+cites", "rrf+cites2hop", "graph+cites2hop"]; report = {}
    for name, path in SETS.items():
        rows = []
        for q in load(path):
            r = engine.retrieve(q["question"], method="all"); L = dict(r["rankings"]); base = L["rrf"]
            view = MaskedAdj(adj, q["gold_ids"]); seeds = base[:3]; cd, c2 = [], []
            for s in seeds:  # mỗi seed một danh sách ứng viên, rồi RRF các seed với nhau
                cd.append(cites_candidates(view, s, None, True)); c2.append(cites_candidates(view, s, None, False))
            cites_d, cites_2 = fuse_rrf(cd, k=60), fuse_rrf(c2, k=60)
            L["rrf+cites"] = fuse_rrf([base, cites_d], k=60); L["graph+cites"] = fuse_rrf([L["graph"], cites_d], k=60)
            L["rrf+cites2hop"] = fuse_rrf([base, cites_2], k=60); L["graph+cites2hop"] = fuse_rrf([L["graph"], cites_2], k=60)
            gold = [q["hop1_id"], q["hop2_id"]] if q["type"] == "bridge" else q["gold_ids"]
            rows.append(dict(id=q["id"], type=q["type"], ranks={s: [rank_of(L[s], g) for g in gold] for s in systems}))
        report[name] = dict(metrics=summarize(rows, systems), mcnemar=[mcnemar(rows, "graph+cites", "graph"), mcnemar(rows, "graph+cites2hop", "graph"), mcnemar(rows, "rrf+cites", "rrf")])
        (out / f"{name}_per_question.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"\n== {name} ==")
        for grp, d in report[name]["metrics"].items():
            print(grp); [print(f"  {s:16s}", {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != 'n'}) for s, v in d.items()]
        for m in report[name]["mcnemar"]: print("  McNemar", m)
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
