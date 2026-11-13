"""Xuất pool ứng viên DEV cho reranker chạy trên GPU (Colab). Offline, không API, không test.

Pool = hợp top-50 của RRF và graph (đúng như scripts/run_rerank_dev.py với MiniLM), để so sánh công bằng.
Đầu ra: results/benchmark_v2_assisted/dev_01/rerank_bge/colab_input.jsonl (mỗi dòng 1 câu hỏi + pool văn bản)
        results/benchmark_v2_assisted/dev_01/rerank_bge/export_meta.json
Chỉ dùng câu hỏi + title/abstract. KHÔNG đưa gold, loại câu hỏi hay thực thể bridge vào đầu vào của reranker.
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPTH = 50


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pool_for(source_row, selected, depth=DEPTH):
    """Hợp có thứ tự (RRF trước, rồi graph) của hai top-depth, bỏ trùng."""
    rankings = source_row["rankings"]
    return list(dict.fromkeys(rankings[selected["rrf"]][:depth] + rankings[selected["graph"]][:depth]))


def input_key(question, pool, texts):
    """Khóa nội dung: Colab trả lại nguyên khóa này, eval đối chiếu để chắc chắn điểm khớp đầu vào."""
    return hashlib.sha256(json.dumps([question, list(zip(pool, texts))], ensure_ascii=False).encode()).hexdigest()


def build(qa, papers, source, selected, depth=DEPTH):
    rows = []
    for q in qa:
        pool = pool_for(source[q["id"]], selected, depth)
        texts = [papers[pid]["title"] + ". " + papers[pid]["abstract"] for pid in pool]
        rows.append(dict(id=q["id"], question=q["question"], key=input_key(q["question"], pool, texts),
                         pool=[dict(id=pid, text=t) for pid, t in zip(pool, texts)]))
    return rows


def main():
    dataset = ROOT / "data/eval/benchmark_v2_assisted"
    base = ROOT / "results/benchmark_v2_assisted/dev_01"
    out = base / "rerank_bge"
    out.mkdir(exist_ok=True)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    corpus = ROOT / manifest["corpus"].replace("\\", "/")
    assert sha(dataset / "qa_dev.jsonl") == manifest["qa_sha256"]["qa_dev.jsonl"], "qa_dev.jsonl đã bị đổi"
    assert sha(corpus) == manifest["corpus_sha256"], "corpus đã bị đổi"
    qa = load(dataset / "qa_dev.jsonl")
    papers = {p["paper_id"]: p for p in load(corpus)}
    source = {r["id"]: r for r in load(base / "retrieval.jsonl")}
    selected = json.loads((base / "selected_config.json").read_text(encoding="utf-8"))
    assert set(source) == {q["id"] for q in qa}
    rows = build(qa, papers, source, selected)
    path = out / "colab_input.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    meta = dict(n_queries=len(rows), n_pairs=sum(len(r["pool"]) for r in rows), depth=DEPTH,
                qa_dev_sha256=sha(dataset / "qa_dev.jsonl"), source_retrieval_sha256=sha(base / "retrieval.jsonl"),
                source_config_sha256=sha(base / "selected_config.json"), colab_input_sha256=sha(path),
                size_mb=round(path.stat().st_size / 1e6, 2), test_access="None. DEV only.",
                scoring="Câu hỏi gốc ghép với title+abstract; mỗi (câu hỏi, văn bản) chấm một lần trên hợp hai pool top-50.")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
