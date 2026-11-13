"""Tạo hai bảng duyệt MÙ (review_R1.xlsx, review_R2.xlsx; cùng thứ tự) từ drafts.jsonl đã qua lọc tự động. Không API.
Người duyệt KHÔNG thấy phán quyết LLM, B_rank hay thông tin hệ thống. Duyệt ĐỘC LẬP, không bàn với nhau cho đến khi nộp.
Chạy: python -X utf8 scripts/make_citation_review_sheet.py [--n 100]"""
import sys, json, random, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import os
D = Path(os.environ["CB_DIR"]) if os.environ.get("CB_DIR") else ROOT / "data/eval/citation_bridge"
GUIDE = ["Mỗi dòng: câu hỏi về paper A mà đáp án nằm ở paper B (abstract). Duyệt ĐỘC LẬP, không hỏi người còn lại.",
         "Cột 'hợp_lệ' = Y khi: (1) câu hỏi tự nhiên, rõ nghĩa, không mơ hồ; (2) đáp án tham chiếu đúng và có thật trong abstract B; (3) câu hỏi không nêu tên/viết tắt riêng của B. Ngược lại = N.",
         "Cột 'A_tự_trả_lời_được' = Y nếu chỉ đọc abstract A (không cần B) cũng trả lời được câu hỏi một cách thực chất; ngược lại = N.",
         "Chỉ dùng nội dung hai abstract; không tra cứu ngoài. Ghi chú nếu cần. Không sửa câu hỏi/đáp án (nếu sai, chấm hợp_lệ = N)."]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); a = ap.parse_args()
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    if (D / "review_R1.xlsx").exists(): sys.exit("Bảng duyệt đã tồn tại; không ghi đè.")
    manifest = json.loads((ROOT / "data/eval/benchmark_v2_assisted/manifest.json").read_text(encoding="utf-8")); corpus = ROOT / manifest["corpus"].replace("\\", "/")
    P = {d["paper_id"]: d for d in (json.loads(l) for l in corpus.read_text(encoding="utf-8").splitlines() if l.strip())}
    drafts = [json.loads(l) for l in (D / "drafts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    good = [x for x in drafts if x["valid"] and x["q"] and not x["leaks_B_name"] and not x["a_only"]
            and x["A_rank"] is not None and x["A_rank"] <= 10]   # câu hỏi phải nhận ra được A (BM25 top-10), nếu không không còn là bridge
    random.Random(20261113).shuffle(good); good = good[:a.n]
    key = {f"r{i + 1:03d}": x for i, x in enumerate(good)}
    (D / "review_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    for rater in ("R1", "R2"):
        wb = Workbook(); g = wb.active; g.title = "Hướng dẫn"
        for i, t in enumerate(GUIDE, 1): g.cell(i, 1, t).alignment = Alignment(wrap_text=True)
        g.column_dimensions["A"].width = 140; ws = wb.create_sheet("Duyệt")
        heads = ["item", "Paper A", "Paper B", "question", "reference_answer", "hợp_lệ", "A_tự_trả_lời_được", "ghi chú"]; widths = [8, 60, 60, 50, 50, 10, 14, 30]
        for c, (h, w) in enumerate(zip(heads, widths), 1):
            cell = ws.cell(1, c, h); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="305496"); ws.column_dimensions[cell.column_letter].width = w
        dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True); ws.add_data_validation(dv)
        for r, (iid, x) in enumerate(key.items(), 2):
            A, B = P[x["A_id"]], P[x["B_id"]]
            vals = [iid, f"[{A['paper_id']}] {A['title']}\n\n{A['abstract']}", f"[{B['paper_id']}] {B['title']}\n\n{B['abstract']}", x["q"], x["ans"], "", "", ""]
            for c, v in enumerate(vals, 1): ws.cell(r, c, v).alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[r].height = 300; dv.add(f"F{r}"); dv.add(f"G{r}")
        ws.freeze_panes = "B2"; wb.save(D / f"review_{rater}.xlsx")
    print(f"{len(key)} mục -> review_R1.xlsx, review_R2.xlsx (khóa ẩn: review_key.json, không đưa cho người duyệt)")


if __name__ == "__main__": main()
