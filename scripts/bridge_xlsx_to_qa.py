"""Đọc bridge_handwritten.xlsx → data/eval/qa_bridge_hand.jsonl (cùng schema qa.jsonl + hop1_id/hop2_id/bridge_entity).
Bỏ dòng không phải Y, thiếu câu hỏi/đáp án, hoặc câu hỏi có chứa bridge_term (không còn là bridge)."""
import re, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from openpyxl import load_workbook


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--xlsx", nargs="+", default=[str(ROOT / "data/eval/bridge_handwritten.xlsx")])
    ap.add_argument("--out", default=str(ROOT / "data/eval/qa_bridge_hand.jsonl")); a = ap.parse_args()
    out, skipped = [], []
    seen = set()
    rowsets = [r for x in a.xlsx for r in load_workbook(x, data_only=True)["Viết tay"].iter_rows(min_row=2, values_only=True)]
    for row in rowsets:
        row = list(row) + [None] * 16; n, t, A, B, ok, q, ans, note = row[:8]; Cc, kind = row[8], row[9]
        if str(ok or "").strip().upper() != "Y": continue
        if not (q and ans): skipped.append((n, "thiếu câu hỏi/đáp án")); continue
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(str(t)) + r"(?![A-Za-z0-9])", str(q), re.I):
            skipped.append((n, f"câu hỏi chứa bridge_term '{t}'")); continue
        ida, idb = re.match(r"\[(.*?)\]", A).group(1), re.match(r"\[(.*?)\]", B).group(1)
        chain = kind == "chain2" and Cc
        idc = re.match(r"\[(.*?)\]", Cc).group(1) if chain else None
        key = (ida, idb, idc)
        if key in seen:
            skipped.append((n, f"trùng cặp {ida}/{idb} đã có ở file trước")); continue
        seen.add(key)
        out.append({"id": f"bh{len(out)+1:03d}", "type": "bridge_chain2" if chain else "bridge", "question": str(q).strip(), "reference_answer": str(ans).strip(),
                    "gold_ids": [ida, idb] + ([idc] if chain else []), "hop1_id": ida, "hop2_id": idc if chain else idb, "bridge_entity": str(t), "group": idc if chain else idb, "source": "handwritten"})
    Path(a.out).write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out), encoding="utf-8")
    print(f"ghi {len(out)} câu -> {a.out}" + (f"\nbỏ qua: {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
