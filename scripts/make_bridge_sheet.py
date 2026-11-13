"""Tạo sheet để viết tay câu bridge từ cặp ứng viên tin cậy cao.  python scripts/make_bridge_sheet.py
Đầu ra: data/eval/bridge_handwritten.xlsx   (viết xong → python scripts/bridge_xlsx_to_qa.py)"""
import json, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

GUIDE = """CÁCH VIẾT 1 CÂU BRIDGE (5–10 câu là đủ cho đợt này)
Mỗi dòng là một cặp (A, B) và một thuật ngữ cầu nối t: paper B GIỚI THIỆU t, paper A DÙNG t.
1) Cột "hợp_lệ": Y nếu đúng là "A dùng thứ B giới thiệu". N nếu không phải. HOMONYM nếu hai bài chỉ trùng tên viết tắt nhưng là hai thứ khác nhau (loại, nhưng ghi chú lại, đây là ca thử tốt cho entity linking).
2) Chỉ viết câu hỏi cho dòng hợp lệ Y.
QUY TẮC CÂU HỎI
 a. Mô tả paper A đủ để nhận ra A (vấn đề, phương pháp của A), KHÔNG nêu tên t (kể cả viết tắt). Nếu t xuất hiện trong câu hỏi thì câu đó không còn là bridge.
 b. Câu hỏi hỏi về điều CHỈ paper B trả lời được (B được xây dựng ra sao, động cơ, kết quả chính của B...). Người chỉ có A không trả lời được; người chỉ có B không biết phải hỏi về A.
 c. Đáp án chỉ lấy từ abstract của B (và A nếu cần để nối). Không dùng kiến thức ngoài hai abstract.
 d. Tránh câu hỏi sẵn cả hai tên paper. Một câu, tự nhiên như người dùng thật hỏi.
VÍ DỤ MẪU (không phải dữ liệu thật): A là "một nghiên cứu theo dõi hiện tượng suy nghĩ quá mức của LLM trên một bộ câu hỏi khoa học trắc nghiệm". Câu hỏi: "Một nghiên cứu đo hiện tượng suy nghĩ quá mức của LLM khi giải các câu hỏi khoa học trắc nghiệm đã dùng một bộ dữ liệu; bộ dữ liệu đó ban đầu được xây dựng ra sao và nhằm đo điều gì?" Đáp án: tóm tắt từ abstract của B.
Điền xong → chạy: python scripts/bridge_xlsx_to_qa.py"""


def main():
    P = {}
    for l in open(ROOT / "data/processed/slice_c1_3000.jsonl", encoding="utf-8"):
        d = json.loads(l); P[d["paper_id"]] = d
    rows = [r for r in csv.DictReader(open(ROOT / "results/analysis/bridge_candidates.csv", encoding="utf-8-sig")) if r["B_introduces"] == "1"]
    wb = Workbook(); g = wb.active; g.title = "Hướng dẫn"
    for i, line in enumerate(GUIDE.split("\n"), 1): g.cell(i, 1, line).alignment = Alignment(wrap_text=True, vertical="top")
    g.column_dimensions["A"].width = 150
    ws = wb.create_sheet("Viết tay")
    heads = ["#", "bridge_term (t)", "Paper A (dùng t)", "Paper B (giới thiệu t)", "hợp_lệ", "question", "reference_answer", "ghi chú"]
    widths = [5, 16, 70, 70, 11, 55, 55, 25]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="305496")
        ws.column_dimensions[cell.column_letter].width = w
    dv = DataValidation(type="list", formula1='"Y,N,HOMONYM"', allow_blank=True); ws.add_data_validation(dv)
    for i, r in enumerate(rows, 1):
        a, b = P[r["A_id"]], P[r["B_id"]]
        vals = [i, r["bridge_term"], f"[{a['paper_id']}] {a['title']}\n\n{a['abstract']}", f"[{b['paper_id']}] {b['title']}\n\n{b['abstract']}"]
        for c, v in enumerate(vals, 1): ws.cell(i + 1, c, v).alignment = Alignment(wrap_text=True, vertical="top")
        for c in (6, 7, 8): ws.cell(i + 1, c).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[i + 1].height = 330; dv.add(f"E{i+1}")
    ws.freeze_panes = "C2"
    out = ROOT / "data/eval/bridge_handwritten.xlsx"; wb.save(out); print(f"ghi {len(rows)} cặp -> {out}")


if __name__ == "__main__":
    main()
