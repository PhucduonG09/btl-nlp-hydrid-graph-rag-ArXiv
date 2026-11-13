"""Tải cạnh trích dẫn A->B (cả hai trong corpus v2) từ Semantic Scholar Graph API (miễn phí, không cần key; có key thì đặt S2_API_KEY).
Đây là nguồn liên kết ĐỘC LẬP với từ ngữ, dùng làm ground truth mức cặp và (tùy chọn) cạnh CITES của KG.
Chạy ở máy có mạng:  pip install requests ;  python -X utf8 scripts/fetch_citations_s2.py
Resume được (cache ở data/external/citations_s2.jsonl). Khoảng 205 request nếu chunk=100, ~1 req/giây => vài phút đến ~20 phút."""
import sys, os, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data.citations import parse_batch
URL = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=externalIds,references.externalIds"
OUT = ROOT / "data/external/citations_s2.jsonl"; CHUNK = 100


def main():
    import requests
    manifest = json.loads((ROOT / "data/eval/benchmark_v2_assisted/manifest.json").read_text(encoding="utf-8"))
    corpus = [json.loads(l)["paper_id"] for l in (ROOT / manifest["corpus"].replace("\\", "/")).read_text(encoding="utf-8").splitlines() if l.strip()]
    cs = set(corpus); done = set()
    if OUT.exists():
        for l in OUT.read_text(encoding="utf-8").splitlines():
            if l.strip(): done.add(json.loads(l)["chunk_first"])
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}
    for i in range(0, len(corpus), CHUNK):
        ids = corpus[i:i + CHUNK]
        if ids[0] in done: continue
        for attempt in range(8):
            r = requests.post(URL, json={"ids": [f"ARXIV:{x}" for x in ids]}, headers=headers, timeout=120)
            if r.status_code == 200: break
            wait = min(60, 2 ** attempt * 2); print(f"HTTP {r.status_code}, thử lại sau {wait}s", flush=True); time.sleep(wait)
        else: sys.exit(f"Thất bại ở chunk {i}: {r.status_code} {r.text[:200]}")
        edges, found = parse_batch(ids, r.json(), cs)
        with OUT.open("a", encoding="utf-8") as f: f.write(json.dumps(dict(chunk_first=ids[0], n_sent=len(ids), n_found=len(found), edges=edges)) + "\n")
        print(f"{i + len(ids)}/{len(corpus)} found {len(found)}/{len(ids)} edges {len(edges)}", flush=True); time.sleep(1.1)
    rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("Tổng: found", sum(r["n_found"] for r in rows), "/", len(corpus), " cạnh", sum(len(r["edges"]) for r in rows))


if __name__ == "__main__": main()
