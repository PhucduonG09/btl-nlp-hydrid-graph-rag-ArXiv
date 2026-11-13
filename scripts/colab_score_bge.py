"""Chấm điểm cross-encoder trên GPU (Colab T4). Script ĐỘC LẬP: không cần repo, chỉ cần colab_input.jsonl.

Dùng:  python colab_score_bge.py --input colab_input.jsonl --output scores.jsonl
Cài:   pip install -q "transformers>=4.40" torch   (Colab đã có sẵn torch)
Chạy tiếp được: nếu Colab ngắt giữa chừng, chạy lại đúng lệnh; các câu hỏi đã có trong --output sẽ được bỏ qua.
Điểm là logit thô của model (đơn điệu với xác suất), chỉ dùng để xếp hạng.
"""
import argparse, json, time
from pathlib import Path


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_rows(rows, predict, done_keys=(), on_record=None):
    """predict(question, texts) -> list[float] cùng độ dài texts. Trả về các record mới."""
    new = []
    for row in rows:
        if row["key"] in done_keys:
            continue
        t = time.perf_counter()
        texts = [c["text"] for c in row["pool"]]
        values = predict(row["question"], texts)
        assert len(values) == len(texts), "số điểm khác số văn bản"
        assert all(v == v and abs(v) != float("inf") for v in values), "điểm NaN/inf"
        rec = dict(id=row["id"], key=row["key"], n_pairs=len(texts), scoring_seconds=time.perf_counter() - t,
                   scores={c["id"]: float(v) for c, v in zip(row["pool"], values)})
        new.append(rec)
        if on_record:
            on_record(rec)
    return new


def make_hf_predict(model_name, revision, max_length, batch_size, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    model = model.half().to(device).eval() if device == "cuda" else model.to(device).eval()
    info = dict(model=model_name, resolved_revision=getattr(model.config, "_commit_hash", None), max_length=max_length,
                batch_size=batch_size, device=device, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)

    @torch.no_grad()
    def predict(question, texts):
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # gom văn bản cùng độ dài để đỡ padding
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([question] * len(idx), [texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            out.extend(zip(idx, model(**enc).logits.view(-1).float().cpu().tolist()))
        return [v for _, v in sorted(out)]
    return predict, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--revision", default=None, help="commit hash để ghim phiên bản; để trống = bản mới nhất, hash thực dùng sẽ được ghi lại")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="chỉ chấm N câu đầu (chạy thử)")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CẢNH BÁO: không thấy GPU. Runtime > Change runtime type > T4 GPU. Chạy CPU sẽ rất chậm.")
    rows = load_jsonl(a.input)[: a.limit]
    done = {r["key"] for r in load_jsonl(a.output)}
    predict, info = make_hf_predict(a.model, a.revision, a.max_length, a.batch_size, device)
    print(json.dumps(info))
    Path(a.output + ".info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    start = time.perf_counter()
    count = [0]

    def write(rec):
        with open(a.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        count[0] += 1
        if count[0] % 10 == 0:
            print(f"{count[0]}/{len(rows) - len(done)} câu, {time.perf_counter() - start:.0f}s", flush=True)

    score_rows(rows, predict, done, write)
    print(f"Xong. Ghi vào {a.output} ({len(load_jsonl(a.output))}/{len(rows)} câu).")


if __name__ == "__main__":
    main()
