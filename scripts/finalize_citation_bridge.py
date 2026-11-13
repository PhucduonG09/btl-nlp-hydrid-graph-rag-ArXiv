"""Chốt bộ bridge trích dẫn từ hai bảng duyệt đã điền. Không API. Nhận câu khi CẢ HAI: hợp_lệ=Y và A_tự_trả_lời=N.
Bất đồng/chưa điền KHÔNG tự phân xử: liệt kê để nhóm họp quyết, rồi sửa tay lại bảng và chạy lại. Đóng băng bằng hash trước khi đánh giá hệ thống.
Chạy: python -X utf8 scripts/finalize_citation_bridge.py [--r1 ...] [--r2 ...]"""
import sys, json, argparse, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.eval.citation_bridge import accept, cohen_kappa
import os
D = Path(os.environ["CB_DIR"]) if os.environ.get("CB_DIR") else ROOT / "data/eval/citation_bridge"


def read(path):
    from openpyxl import load_workbook
    out = {}
    for r in load_workbook(path)["Duyệt"].iter_rows(min_row=2, values_only=True):
        if r[0]: out[r[0]] = (str(r[5]).strip().upper() if r[5] else "", str(r[6]).strip().upper() if r[6] else "")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r1", default=str(D / "review_R1.xlsx")); ap.add_argument("--r2", default=str(D / "review_R2.xlsx")); a = ap.parse_args()
    key = json.loads((D / "review_key.json").read_text(encoding="utf-8")); R1, R2 = read(a.r1), read(a.r2)
    status = {i: accept(R1.get(i, ("", "")), R2.get(i, ("", ""))) for i in key}; counts = {s: sum(v == s for v in status.values()) for s in ("accepted", "rejected", "disagreement", "pending")}
    both = [i for i in key if status[i] != "pending"]
    agree = dict(n_rated_by_both=len(both), kappa_valid=cohen_kappa([R1[i][0] for i in both], [R2[i][0] for i in both]),
                 kappa_A_alone=cohen_kappa([R1[i][1] for i in both], [R2[i][1] for i in both]),
                 pct_agree_valid=round(sum(R1[i][0] == R2[i][0] for i in both) / max(len(both), 1), 3))
    print(counts, agree)
    if counts["pending"] or counts["disagreement"]:
        print("Chưa chốt được. Bất đồng:", [i for i, s in status.items() if s == "disagreement"], "| chưa điền:", [i for i, s in status.items() if s == "pending"]); return
    acc = [i for i in key if status[i] == "accepted"]
    if len(acc) < 40: print(f"CẢNH BÁO: chỉ {len(acc)} câu được nhận (<40): khoảng tin cậy sẽ rộng; ghi rõ trong báo cáo.")
    qa, cov = [], {}
    for n, i in enumerate(acc, 1):
        x = key[i]; qid = f"cbq{n:03d}"
        qa.append(dict(id=qid, type="bridge", question=x["q"], reference_answer=x["ans"], gold_ids=[x["A_id"], x["B_id"]], hop1_id=x["A_id"], hop2_id=x["B_id"],
                       origin="citation_natural", source="semantic_scholar_citation_A_cites_B", pair_id=x["pair_id"]))
        cov[qid] = dict(B_rank_bm25=x["B_rank"], A_rank_bm25=x["A_rank"], shares_mentions_entity=x["covariate_shares_mentions_entity"])
    p = D / "qa_citation_bridge.jsonl"
    if p.exists(): sys.exit("qa_citation_bridge.jsonl đã tồn tại (đã đóng băng); không ghi đè.")
    p.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qa), encoding="utf-8")
    (D / "covariates.json").write_text(json.dumps(cov, indent=1), encoding="utf-8")
    (D / "frozen.json").write_text(json.dumps(dict(n=len(qa), qa_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), agreement=agree, counts=counts,
                                                   note="Đóng băng TRƯỚC khi chạy bất kỳ hệ thống nào trên bộ này."), indent=2), encoding="utf-8"); print(f"Đã chốt {len(qa)} câu.")


if __name__ == "__main__": main()
