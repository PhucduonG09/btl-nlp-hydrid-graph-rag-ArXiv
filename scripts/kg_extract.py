"""Trích xuất quan hệ bằng LLM.
Ví dụ:
  python scripts/kg_extract.py --input data/eval/dev_extract.jsonl --out data/kg/dev_extract_out_v3.jsonl --prompt v3
  python scripts/kg_extract.py --input data/processed/pilot_papers.jsonl --out data/kg/pilot_extract_v3.jsonl --prompt v3 --workers 6
Chạy lại cùng lệnh sẽ tiếp tục phần còn thiếu (resume)."""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kg.llm_extract import run_extraction
from src.common.llm import cost_summary

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="v3")
ap.add_argument("--model", default=None)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--json-mode", action="store_true")
ap.add_argument("--max-tokens", type=int, default=None)
a = ap.parse_args()

papers = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
if a.limit:
    papers = papers[:a.limit]
t0 = time.time()
outs, errors = run_extraction(papers, a.out, a.prompt, a.model, a.workers, a.json_mode, a.max_tokens)
dt = time.time() - t0
if errors:
    print(f"\n{len(errors)} lỗi, ví dụ: {errors[0]}")
    print("Chạy lại cùng lệnh để thử tiếp các paper còn thiếu.")
n_err = sum("error" in o for o in outs)
n_rel = sum(len(o["relations"]) for o in outs)
print(f"\n{len(outs)}/{len(papers)} paper | {n_rel} quan hệ | JSON lỗi: {n_err} | "
      f"thiếu evidence bị loại: {sum(o.get('dropped_no_evidence', 0) for o in outs)} | {dt:.0f}s")
print("Chi phí lượt này:", cost_summary([o["usage"] for o in outs if o["usage"].get("cached") is False]))
