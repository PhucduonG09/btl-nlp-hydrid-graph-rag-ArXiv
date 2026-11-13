import json, os

def done_ids(path):
    """Đọc các id đã xử lý từ file JSONL output (resume sau khi mất phiên)."""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(l)["paper_id"] for l in f if l.strip()}

def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())