"""Phân tích phản hồi Semantic Scholar batch -> cạnh trích dẫn trong corpus (A cites B)."""


def parse_batch(ids, data, corpus_ids):
    """ids: list arXiv id đã gửi (cùng thứ tự); data: list kết quả (None nếu không tìm thấy).
    Trả (edges, found) với edges = [(A, B)] sao cho cả hai thuộc corpus_ids và A != B."""
    edges, found = [], set()
    for pid, item in zip(ids, data):
        if not item: continue
        found.add(pid)
        for ref in item.get("references") or []:
            ext = (ref or {}).get("externalIds") or {}
            b = ext.get("ArXiv")
            if b and b in corpus_ids and b != pid: edges.append((pid, b))
    return sorted(set(edges)), found
