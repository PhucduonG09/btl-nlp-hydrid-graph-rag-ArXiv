"""Tạo file Excel gán nhãn cho 2 người từ lát cắt. Gán DỄ + đánh tag; quy tắc áp dụng sau bằng bộ lọc tag.
  python scripts/make_annotation_sheet.py --slice data/processed/slice_c1_3000.jsonl --n 60 --overlap 30
Đầu ra: data/eval/annotation/annotator_A.xlsx, annotator_B.xlsx, manifest.json
  - `overlap` paper được CẢ HAI gán độc lập (để đo đồng thuận); phần còn lại chia đôi.
  - Loại sẵn 39 paper đã gán ở dev/test (đã dùng để tune prompt)."""
import json, random, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

RELS = "PROPOSES,USES,EVALUATES_ON,ADDRESSES,NONE"
TAGS = "GENERIC,UNNAMED,BASELINE,STUDIED,OTHER"
GUIDE = """HƯỚNG DẪN GÁN NHÃN (v4-nháp) — nguyên tắc: GÁN DỄ, ĐÁNH TAG. Đừng tự xoá vì nghĩ "chắc không cần".
Mỗi quan hệ = 1 dòng: relation | target | evidence | tag | note. Chủ thể luôn là paper. Mỗi paper có sẵn 8 dòng trống; dư thì để trống.

RELATION
- PROPOSES: cái MỚI do paper đưa ra (method, model, framework, dataset/benchmark mới). Tên tác giả đặt (AdaFish, C2G); nếu không có tên: cụm ≤6 từ ngắn nhất + tag UNNAMED.
- USES: thứ ĐÃ CÓ được dùng làm thành phần/công cụ/nguồn dữ liệu trong pipeline của chính paper (UMAP, SPARQL, Wikipedia, Qwen3).
- EVALUATES_ON: dataset/benchmark ĐÃ CÓ, chỉ dùng để đánh giá (HotpotQA, Spider, BIRD).  [mặc định, nhóm có thể đổi]
- ADDRESSES: bài toán/tác vụ paper giải quyết (1–2 mục chính).
- NONE: paper KHÔNG có PROPOSES/USES/EVALUATES_ON nào (kể cả USES+STUDIED) — thường là khảo sát, position paper. Ghi 1 dòng NONE (lý do ở cột note) và VẪN gán ADDRESSES (chủ đề).
  Taxonomy/phân loại/kết luận/danh sách thách thức của survey KHÔNG phải PROPOSES (không phải method/model/framework/dataset).

EVIDENCE: copy NGUYÊN VĂN từ tiêu đề/abstract (câu hoặc đoạn ngắn). Bắt buộc, kể cả khi có tên.

TAG (để trống nếu không dính; dính nhiều tag thì ghi cách nhau dấu ;)
- GENERIC: khái niệm chung gần như paper nào cũng nhắc (LLM, deep learning, Transformer, CNN, MLP, gradient descent...).
- UNNAMED: không có tên riêng, bạn phải tự đặt/ghi mô tả.
- BASELINE: chỉ để so sánh, nền, related work, ví dụ minh hoạ — KHÔNG phải thành phần pipeline của paper. (Vẫn ghi lại nếu phân vân!)
- STUDIED: phương pháp/mô hình là ĐỐI TƯỢNG bị khảo sát/so sánh trong nghiên cứu thực nghiệm, không phải thành phần xây dựng.
- OTHER: bạn phân vân vì lý do khác; ghi rõ ở note.
KHÔNG đọc dự đoán của extractor trong khi gán (tránh thiên lệch). Gán xong mới trao đổi với người kia."""


def build(path, papers, slots):
    wb = Workbook(); g = wb.active; g.title = "Hướng dẫn"
    for i, line in enumerate(GUIDE.split("\n"), 1):
        g.cell(i, 1, line).alignment = Alignment(wrap_text=True, vertical="top")
    g.column_dimensions["A"].width = 140
    ws = wb.create_sheet("Gán nhãn")
    heads = ["paper_id", "title & abstract", "relation", "target", "evidence", "tag", "note"]
    widths = [14, 90, 15, 30, 50, 18, 30]
    for c, (h, w) in enumerate(zip(heads, widths), 1):
        cell = ws.cell(1, c, h); cell.font = Font(bold=True, color="FFFFFF"); cell.fill = PatternFill("solid", fgColor="305496")
        ws.column_dimensions[cell.column_letter].width = w
    dv_rel = DataValidation(type="list", formula1=f'"{RELS}"', allow_blank=True)
    dv_tag = DataValidation(type="list", formula1=f'"{TAGS}"', allow_blank=True, showErrorMessage=False)
    ws.add_data_validation(dv_rel); ws.add_data_validation(dv_tag)
    r = 2
    for k, p in enumerate(papers):
        fill = PatternFill("solid", fgColor="F2F2F2" if k % 2 == 0 else "FFFFFF")
        top, bot = r, r + slots - 1
        for rr in range(top, bot + 1):
            ws.cell(rr, 1, p["paper_id"]); ws.row_dimensions[rr].height = 48
            for c in range(1, 8): ws.cell(rr, c).fill = fill
            ws.cell(rr, 5).alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=top, end_row=bot, start_column=2, end_column=2)
        ws.cell(top, 2, f"{p['title']}\n\n{p['abstract']}").alignment = Alignment(wrap_text=True, vertical="top")
        dv_rel.add(f"C{top}:C{bot}"); dv_tag.add(f"F{top}:F{bot}")
        r = bot + 1
    ws.freeze_panes = "C2"
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default=str(ROOT / "data/processed/slice_c1_3000.jsonl"))
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--overlap", type=int, default=30)
    ap.add_argument("--slots", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=str(ROOT / "data/eval/annotation"))
    a = ap.parse_args()
    sl = [json.loads(l) for l in open(a.slice, encoding="utf-8") if l.strip()]
    used = {json.loads(l)["paper_id"] for f in ("dev", "test")
            for l in open(ROOT / f"data/eval/{f}_extract.jsonl", encoding="utf-8") if l.strip()}
    cand = [p for p in sl if p["paper_id"] not in used]
    rng = random.Random(a.seed); pick = rng.sample(cand, a.n)
    both, rest = pick[:a.overlap], pick[a.overlap:]
    half = len(rest) // 2
    A = both + rest[:half]; B = both + rest[half:]
    rng.shuffle(A); rng.shuffle(B)          # thứ tự khác nhau → không biết paper nào là phần chung
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    build(out / "annotator_A.xlsx", A, a.slots); build(out / "annotator_B.xlsx", B, a.slots)
    man = {"seed": a.seed, "slice": a.slice, "overlap": [p["paper_id"] for p in both],
           "only_A": [p["paper_id"] for p in rest[:half]], "only_B": [p["paper_id"] for p in rest[half:]]}
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(f"ứng viên {len(cand)} (đã loại {len(sl) - len(cand)} paper đã gán cũ); chọn {a.n}: chung {len(both)}, "
          f"A riêng {half}, B riêng {len(rest) - half}\nA gán {len(A)} paper, B gán {len(B)} paper -> {out}")


if __name__ == "__main__":
    main()
