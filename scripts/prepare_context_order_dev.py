"""Chuẩn bị thí nghiệm thứ tự/độ dài ngữ cảnh trên DEV bridge. Offline, KHÔNG gọi API, không test.

Giả thuyết (khai báo TRƯỚC khi chạy): với hệ graph, ngữ cảnh 10 abstract (~3K token) đủ ngắn để LLM mạnh
không bị 'lost in the middle'; đổi thứ tự (reverse, sandwich) không làm đổi số câu đúng quá mức nhiễu.
Chỉ biến thể top5 dự kiến GIẢM (paper B ở hạng 3-9 nên nhiều câu mất B).
Trần cải thiện: baseline graph bridge có 23 đúng, 3 một phần, 5 thiếu gold trong ngữ cảnh -> đổi thứ tự tối đa +3 câu.
Quy tắc kết luận (khai báo trước): chỉ nói 'có hiệu ứng vị trí' nếu McNemar exact p<0.05 VÀ chênh >=4 câu; n=31 nên
chỉ phát hiện được hiệu ứng lớn, không có nghĩa là 'không có hiệu ứng nhỏ'.
Kiểm soát biến: cùng retrieval (graph), cùng 31 câu, cùng prompt/model/ngân sách; reverse/sandwich dùng đúng tập paper
của baseline top-10, chỉ đổi thứ tự (và số thứ tự [n] theo vị trí mới).
"""
import hashlib, json, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generation.controlled import prepare, VERSION
from src.generation.ordering import reorder

BASE = ROOT / "results/benchmark_v2_assisted/dev_01/generation_prepared"
OUT = Path(os.environ.get("CTX_ORDER_DIR", BASE / "context_order"))
MODES = ["reverse", "sandwich", "top5"]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean_pos(order, ids):
    return [order.index(i) + 1 if i in order else None for i in ids]


def main():
    os.environ["TIKTOKEN_CACHE_DIR"] = str(ROOT / "data/cache/tiktoken")
    import tiktoken
    encoder = tiktoken.get_encoding("cl100k_base")
    dataset = ROOT / "data/eval/benchmark_v2_assisted"
    dev = ROOT / "results/benchmark_v2_assisted/dev_01"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    corpus = ROOT / manifest["corpus"].replace("\\", "/")
    assert sha(dataset / "qa_dev.jsonl") == manifest["qa_sha256"]["qa_dev.jsonl"]
    assert sha(corpus) == manifest["corpus_sha256"]
    qa = [q for q in load(dataset / "qa_dev.jsonl") if q["type"] == "bridge"]
    papers = {p["paper_id"]: p for p in load(corpus)}
    runs = {r["id"]: r for r in load(dev / "retrieval.jsonl")}
    cfg = json.loads((dev / "selected_config.json").read_text(encoding="utf-8"))
    baseline = {j["id"]: j for j in load(BASE / "jobs_dev.jsonl") if j["system"] == "graph"}
    OUT.mkdir(parents=True, exist_ok=True)
    jobs, checks, positions = [], dict(rank_prompt_identical=0, same_paper_set=0, truncated_sources=0), []
    for q in qa:
        ranking = runs[q["id"]]["rankings"][cfg["graph"]][:10]
        control = prepare(q["question"], reorder(ranking, "rank"), papers, encoder, budget=6000, top_k=10)
        assert control["messages"] == baseline[q["id"]]["messages"], "pipeline không tái tạo đúng prompt baseline"
        checks["rank_prompt_identical"] += 1
        for mode in MODES:
            order = reorder(ranking, mode)
            payload = prepare(q["question"], order, papers, encoder, budget=6000, top_k=len(order))
            checks["truncated_sources"] += sum(s["truncated"] for s in payload["sources"])
            if mode != "top5":
                assert {s["paper_id"] for s in payload["sources"]} == {s["paper_id"] for s in baseline[q["id"]]["sources"]}
                checks["same_paper_set"] += 1
            jobs.append(dict(id=q["id"], system=f"graph_{mode}", retrieval_method=cfg["graph"], order=mode,
                             order_paper_ids=[s["paper_id"] for s in payload["sources"]], **payload))
            positions.append(dict(id=q["id"], mode=mode, gold_positions=mean_pos([s["paper_id"] for s in payload["sources"]], q["gold_ids"]),
                                  n_sources=len(payload["sources"])))
    assert checks["truncated_sources"] == 0, "ngân sách token cắt abstract: không còn là thay đổi thuần thứ tự"
    rng = random.Random(42)
    pilot_ids = sorted(q["id"] for q in rng.sample(qa, 2))
    pilot = [j for j in jobs if j["id"] in pilot_ids]
    for name, rows in [("jobs_dev.jsonl", jobs), ("pilot_jobs.jsonl", pilot)]:
        (OUT / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "positions.json").write_text(json.dumps(positions, indent=1), encoding="utf-8")
    plan = dict(status="PREPARED_NOT_EXECUTED", experiment="context_order_dev_bridge", system="graph", modes=MODES,
                baseline_reused="claude5_controlled (system=graph, mode=rank): cùng prompt, đã kiểm tra bằng assert", n_bridge_questions=len(qa),
                n_jobs=len(jobs), pilot_question_ids=pilot_ids, pilot_jobs=len(pilot), prompt_version=VERSION,
                qa_dev_sha256=sha(dataset / "qa_dev.jsonl"), jobs_sha256=sha(OUT / "jobs_dev.jsonl"), pilot_jobs_sha256=sha(OUT / "pilot_jobs.jsonl"),
                dev_input_proxy_tokens=sum(j["input_content_proxy_tokens"] for j in jobs), dev_max_output_tokens=len(jobs) * 768,
                checks=checks, paid_calls=0, test_access="None. DEV only; không mở test.",
                ceiling="baseline graph bridge: 23 correct, 3 partial, 5 thiếu gold trong ngữ cảnh -> đổi thứ tự tối đa +3 câu",
                decision_rule="chỉ nói có hiệu ứng vị trí nếu McNemar exact p<0.05 và chênh >=4 câu; n=31, judge tự động chưa hiệu chuẩn",
                judge_note="judge phải khác họ model với generator (runner kiểm tra); dùng cùng LLM_MODEL/LLM_JUDGE_MODEL như lần chạy baseline")
    (OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: plan[k] for k in ["status", "n_jobs", "pilot_question_ids", "dev_input_proxy_tokens", "checks", "paid_calls"]}, indent=2))


if __name__ == "__main__":
    main()
