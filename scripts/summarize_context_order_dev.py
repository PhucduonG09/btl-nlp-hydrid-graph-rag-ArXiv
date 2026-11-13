"""Tổng hợp thí nghiệm thứ tự ngữ cảnh (offline). So từng biến thể với baseline graph-rank trên câu bridge DEV.
Điểm là của judge tự động CHƯA hiệu chuẩn; kết quả thăm dò trên DEV.

Quy tắc loại câu (không phụ thuộc đúng/sai): câu bị loại khỏi MỌI hệ thống nếu bất kỳ hệ thống nào thiếu đáp án hoặc
judge hợp lệ (vd judge suy biến, lỗi JSON), để so sánh ghép cặp trên cùng một tập câu. --drop-mismatch loại thêm các câu có
đáp án bị phục vụ bằng model khác nhãn yêu cầu (mặc định chỉ cảnh báo)."""
import argparse, json, os, sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.answer_metrics import latest_records, summarize_answers

BASE = ROOT / "results/benchmark_v2_assisted/dev_01/generation_prepared"
OUT = Path(os.environ.get("CTX_ORDER_DIR", BASE / "context_order"))
MODES = ["reverse", "sandwich", "top5"]


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mcnemar_exact(a_only, b_only):
    n = a_only + b_only
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n)


def valid(g, j, i, key):
    return (i, key) in g and not g[(i, key)].get("error") and (i, key) in j and not j[(i, key)].get("error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-mismatch", action="store_true")
    args = ap.parse_args()
    qa = {q["id"]: q for q in load(ROOT / "data/eval/benchmark_v2_assisted/qa_dev.jsonl") if q["type"] == "bridge"}
    run = BASE / "claude5_controlled"
    gen = latest_records(load(run / "generation.jsonl")); jud = latest_records(load(run / "judge.jsonl"))
    new_gen = latest_records(load(OUT / "generation.jsonl")); new_jud = latest_records(load(OUT / "judge.jsonl"))
    systems = {"graph_rank": (gen, jud, "graph")}
    for mode in MODES:
        systems[f"graph_{mode}"] = (new_gen, new_jud, f"graph_{mode}")
    excluded = {}
    for i in sorted(qa):
        bad = [n for n, (g, j, k) in systems.items() if not valid(g, j, i, k)]
        if args.drop_mismatch:
            bad += [n + ":label_mismatch" for n, (g, j, k) in systems.items() if valid(g, j, i, k) and g[(i, k)].get("model_label_mismatch")]
        if bad:
            excluded[i] = bad
    ids = [i for i in sorted(qa) if i not in excluded]
    mismatched = [(i, n, g[(i, k)].get("served_model")) for i in ids for n, (g, j, k) in systems.items() if g[(i, k)].get("model_label_mismatch")]
    if excluded:
        print(f"Loại {len(excluded)} câu khỏi mọi hệ thống (thiếu đáp án/judge hợp lệ): {excluded}")
    if mismatched:
        print(f"CẢNH BÁO: {len(mismatched)} đáp án dùng sai nhãn model, vẫn được tính: {mismatched}")
        print("Chạy scripts/flag_label_mismatch.py rồi --retry-errors, hoặc dùng --drop-mismatch, hoặc báo cáo kèm cảnh báo.")
    summary, correct = {}, {}
    for name, (g, j, key) in systems.items():
        summary[name] = summarize_answers([g[(i, key)] for i in ids], [j[(i, key)] for i in ids])
        correct[name] = {i: j[(i, key)]["quality"] == "correct" for i in ids}
    paired = {}
    for name in [f"graph_{m}" for m in MODES]:
        a = [i for i in ids if correct["graph_rank"][i] and not correct[name][i]]
        b = [i for i in ids if correct[name][i] and not correct["graph_rank"][i]]
        p = mcnemar_exact(len(a), len(b))
        paired[name] = dict(rank_only_correct=len(a), variant_only_correct=len(b), p_exact=p, flipped_to_wrong=a, flipped_to_correct=b,
                            claims_position_effect=bool(p < .05 and abs(len(a) - len(b)) >= 4))
    report = dict(n_bridge_used=len(ids), n_bridge_total=len(qa), excluded=excluded, drop_mismatch=args.drop_mismatch,
                  n_label_mismatch_used=len(mismatched), label_mismatches=mismatched, systems=summary, paired_vs_rank=paired,
                  exploratory=True, test_evaluated=False,
                  notes="Judge tự động chưa hiệu chuẩn; trần cải thiện +3 câu; quy tắc kết luận khai báo trong plan.json.")
    (OUT / "context_order_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nn dùng = {len(ids)}/{len(qa)} câu bridge")
    print(f"{'system':16s} {'correct':>8s} {'partial':>8s} {'abstain':>8s} {'incorr':>7s} {'faith':>6s} {'relev':>6s}")
    for name, s in summary.items():
        qc = s["quality_counts"]
        print(f"{name:16s} {qc.get('correct', 0):8d} {qc.get('partial', 0):8d} {qc.get('abstained', 0):8d} {qc.get('incorrect', 0):7d} "
              f"{(s['faithfulness_macro_nonabstaining'] or 0):6.3f} {s['mean_relevance']:6.2f}")
    for name, p in paired.items():
        print(f"{name}: rank-only={p['rank_only_correct']} variant-only={p['variant_only_correct']} p={p['p_exact']:.3f} "
              f"hiệu_ứng_vị_trí={p['claims_position_effect']}  (→sai: {p['flipped_to_wrong']}; →đúng: {p['flipped_to_correct']})")


if __name__ == "__main__":
    main()
