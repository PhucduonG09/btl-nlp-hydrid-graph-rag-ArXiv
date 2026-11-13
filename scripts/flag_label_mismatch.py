"""Đánh dấu để chạy lại các job bị gateway phục vụ bằng model khác nhãn yêu cầu (vd claude-sonnet-5-ta). Offline, không API.

Log là append-only và bản ghi MỚI NHẤT thắng, nên script chỉ THÊM dòng đánh dấu lỗi (giữ nguyên bản ghi cũ để kiểm toán):
  - generation.jsonl: error='served_label_mismatch_regenerate' cho job lệch nhãn
  - judge.jsonl:      error='generation_regenerated_rejudge' cho cùng job (vì đáp án sẽ đổi)
Sau đó chạy lại các stage với --retry-errors (runner bỏ qua cache khi retry). Có thể lặp lại nếu lần chạy lại vẫn lệch nhãn.
  python scripts/flag_label_mismatch.py [--dir results/benchmark_v2_assisted/dev_01/generation_prepared/context_order] [--dry-run]
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/benchmark_v2_assisted/dev_01/generation_prepared/context_order"


def load(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def latest(rows):
    return {(r["id"], r["system"]): r for r in rows}


def flag(directory, dry_run=False):
    d = Path(directory)
    gen = latest(load(d / "generation.jsonl"))
    keys = sorted(k for k, r in gen.items() if r.get("model_label_mismatch") and not r.get("error"))
    gen_marks = [dict(id=k[0], system=k[1], type=gen[k].get("type"), error="served_label_mismatch_regenerate",
                      superseded_served_model=gen[k].get("served_model")) for k in keys]
    judge_marks = [dict(id=k[0], system=k[1], type=gen[k].get("type"), error="generation_regenerated_rejudge") for k in keys]
    if keys and not dry_run:
        for name, rows in [("generation.jsonl", gen_marks), ("judge.jsonl", judge_marks)]:
            with (d / name).open("a", encoding="utf-8") as f:
                f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return keys


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    keys = flag(a.dir, a.dry_run)
    print(f"{'(dry-run) ' if a.dry_run else ''}đánh dấu {len(keys)} job lệch nhãn để chạy lại: {[f'{i}/{s}' for i, s in keys][:20]}")
