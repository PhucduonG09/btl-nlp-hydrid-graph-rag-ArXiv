"""Cạnh CITES thật (Semantic Scholar) có bổ sung cho truy xuất hybrid khi GIẤU chính cạnh (A,B) cần dự đoán? (leave-one-edge-out)
Seed A; ứng viên = các paper A trích dẫn khác B + láng giềng 2 bước (đồ thị vô hướng, bỏ cạnh A-B). Điểm: 2*[c láng giềng trực tiếp] + sum_x 1/log(2+deg x).
Không tune: hai trọng số fusion khai báo trước [1,1] và [1,2] (base:cites). Quy tắc thắng: Hit@10 (RRF base+cites) - base có CI bootstrap 95% > 0.
Lưu ý: đây là dự đoán liên kết trích dẫn (liên quan chủ đề/bibliographic), KHÔNG phải QA multi-hop; ground truth trích dẫn mang nhiễu (tham chiếu nền như BERT/Adam).
Chạy: python -X utf8 scripts/citation_cites_leaveout.py"""
import sys, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.retrieval.graph import fuse_rrf
from scripts.citation_gating import boot, hit


def cites_candidates(adj, a, b, include_direct=True):
    direct = adj[a] - {b}; score = defaultdict(float)
    for c in (direct if include_direct else ()): score[c] += 2
    for x in direct:
        w = 1 / math.log(2 + len(adj[x]))
        for c in adj[x]:
            if c != a: score[c] += w   # B được phép xuất hiện qua đường a-x-b; chỉ cạnh trực tiếp a-b bị giấu
    score.pop(a, None)
    if not include_direct:
        for c in direct: score.pop(c, None)
    return [c for c, _ in sorted(score.items(), key=lambda t: (-t[1], t[0]))][:100]


def main():
    edges = sorted({tuple(e) for l in (ROOT / "data/external/citations_s2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip() for e in json.loads(l)["edges"]})
    adj = defaultdict(set)
    for x, y in edges: adj[x].add(y); adj[y].add(x)
    rows = json.loads((ROOT / "results/analysis/citation_lists_4000.json").read_text(encoding="utf-8"))
    base_h, w11, w12, reach, has_other, t11, t12, treach = [], [], [], [], [], [], [], []
    for r in rows:
        a, b = r["A"], r["B"]; base = fuse_rrf([r["bm25"], r["dense"]], k=10); cand = cites_candidates(adj, a, b)
        # B không được đi qua chính cạnh (a,b): cites_candidates chỉ dùng direct-{b}; đường a-x-b dùng cạnh x-b hợp lệ.
        c2 = cites_candidates(adj, a, b, include_direct=False); treach.append(b in c2)
        t11.append(hit(fuse_rrf([base, c2], k=60, weights=[1, 1]), b)); t12.append(hit(fuse_rrf([base, c2], k=60, weights=[1, 2]), b))
        base_h.append(hit(base, b)); has_other.append(len(adj[a] - {b}) > 0); reach.append(b in cand)
        w11.append(hit(fuse_rrf([base, cand], k=60, weights=[1, 1]), b)); w12.append(hit(fuse_rrf([base, cand], k=60, weights=[1, 2]), b))
    base_h, w11, w12, reach, has_other, t11, t12, treach = [np.array(v, float) for v in (base_h, w11, w12, reach, has_other, t11, t12, treach)]
    m = has_other.astype(bool)
    res = dict(n=len(rows), share_A_has_other_edges=round(float(has_other.mean()), 3), B_reachable_in_cites_top100=round(float(reach.mean()), 3), base_hit10=round(float(base_h.mean()), 4),
               all_pairs={"w[1,1]": dict(hit10=round(float(w11.mean()), 4), diff_vs_base=boot(w11 - base_h)), "w[1,2]": dict(hit10=round(float(w12.mean()), 4), diff_vs_base=boot(w12 - base_h))},
               only_A_has_other_edges=dict(n=int(m.sum()), base=round(float(base_h[m].mean()), 4), **{"w[1,1]": boot(w11[m] - base_h[m]), "w[1,2]": boot(w12[m] - base_h[m])}),
               EXPLORATORY_twohop_only_posthoc=dict(note="Thêm SAU khi thấy kết quả bản có láng giềng trực tiếp (lỗi thiết kế: tham chiếu trực tiếp lấn át 2 bước); không phải xác nhận.",
                   B_reachable_top100=round(float(treach.mean()), 3), **{"w[1,1]": dict(hit10=round(float(t11.mean()), 4), diff_vs_base=boot(t11 - base_h)), "w[1,2]": dict(hit10=round(float(t12.mean()), 4), diff_vs_base=boot(t12 - base_h))}),
               note="Leave-one-edge-out; ground truth là trích dẫn, không phải câu hỏi multi-hop. Cạnh CITES là dữ liệu ngoài (S2), không suy ra từ từ vựng.")
    (ROOT / "results/analysis/citation_cites_leaveout.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
