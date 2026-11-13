"""Đánh giá 5 phương pháp ĐÃ KHÓA trên bộ bridge trích dẫn đã đóng băng (docs/protocol-citation-bridge.md). Không API.
Chạy MỘT lần: python -X utf8 scripts/eval_citation_bridge.py --run-id cb_01   (cần BGE local, env_AI)"""
import sys, json, argparse, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.eval_two_hop_baseline import load, rank_of, summarize, mcnemar
D = ROOT / "data/eval/citation_bridge"; METHODS = ["bm25", "dense", "rrf", "text_expansion", "graph"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", required=True); a = ap.parse_args()
    frozen = json.loads((D / "frozen.json").read_text(encoding="utf-8")); p = D / "qa_citation_bridge.jsonl"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == frozen["qa_sha256"], "Bộ câu hỏi đã bị sửa sau khi đóng băng"
    receipt = D / "evaluated.json"
    if receipt.exists(): sys.exit("Đã đánh giá một lần (xem evaluated.json). Chạy lại chỉ được coi là thăm dò, hãy ghi rõ.")
    out = ROOT / "results/citation_bridge" / a.run_id
    if out.exists(): sys.exit("run-id đã tồn tại.")
    out.mkdir(parents=True); qa = load(p); cov = json.loads((D / "covariates.json").read_text(encoding="utf-8"))
    from src.retrieval.engine import load_engine
    engine = load_engine(); rows = []
    for i, q in enumerate(qa, 1):
        L = engine.retrieve(q["question"], method="all")["rankings"]
        rows.append(dict(id=q["id"], type="bridge", ranks={m: [rank_of(L[m], g) for g in (q["hop1_id"], q["hop2_id"])] for m in METHODS}, **cov[q["id"]]))
        if i % 10 == 0 or i == len(qa): print(f"{i}/{len(qa)}", flush=True)
    (out / "per_question.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    strata = {"all": rows, "B_rank_bm25<=10": [r for r in rows if r["B_rank_bm25"] and r["B_rank_bm25"] <= 10], "B_rank_bm25>10": [r for r in rows if not (r["B_rank_bm25"] and r["B_rank_bm25"] <= 10)],
              "shares_mentions_entity": [r for r in rows if r["shares_mentions_entity"]], "no_shared_mentions_entity": [r for r in rows if not r["shares_mentions_entity"]]}
    rep = {k: dict(n=len(v), metrics=summarize(v, METHODS)["bridge"] if v else None) for k, v in strata.items()}
    hard = strata["B_rank_bm25>10"]; rep["mcnemar_primary_hard_stratum"] = dict(n=len(hard), exploratory_if_n_lt_25=len(hard) < 25, **(mcnemar(hard, "graph", "rrf") if hard else {}))
    rep["mcnemar_primary"] = mcnemar(rows, "graph", "rrf"); rep["mcnemar_secondary"] = [mcnemar(rows, "graph", "dense"), mcnemar(rows, "text_expansion", "rrf")]
    rep["frozen"] = frozen; (out / "metrics.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt.write_text(json.dumps(dict(run_id=a.run_id, qa_sha256=frozen["qa_sha256"]), indent=2), encoding="utf-8")
    for k, v in rep.items():
        if isinstance(v, dict) and "metrics" in v and v["metrics"]:
            print(f"\n== {k} (n={v['n']}) =="); [print(f"  {m:15s}", {x: (round(y, 3) if isinstance(y, float) else y) for x, y in d.items() if x != 'n'}) for m, d in v["metrics"].items()]
    print("\nMcNemar CHÍNH (tầng B_rank>10):", rep["mcnemar_primary_hard_stratum"]); print("McNemar toàn bộ:", rep["mcnemar_primary"]); [print("McNemar phụ:", m) for m in rep["mcnemar_secondary"]]


if __name__ == "__main__": main()
