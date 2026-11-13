"""Logic thuần cho bộ bridge trích dẫn: quy tắc nhận câu và độ đồng thuận giữa hai người duyệt."""


def cohen_kappa(a, b):
    """Kappa cho hai chuỗi nhãn cùng độ dài (nhãn bất kỳ). Trả None nếu không xác định (mẫu rỗng hoặc chỉ một nhãn)."""
    n = len(a)
    if n == 0: return None
    labels = sorted(set(a) | set(b)); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


def accept(r1, r2):
    """r = (hợp_lệ, A_tự_trả_lời) với 'Y'/'N'. Nhận khi cả hai: hợp lệ=Y và A_tự_trả_lời=N.
    Trả 'accepted' | 'rejected' | 'disagreement' | 'pending'. Bất đồng KHÔNG tự phân xử: ghi lại để nhóm họp quyết (nhật ký công khai)."""
    if None in r1 or None in r2 or "" in r1 or "" in r2: return "pending"
    ok = lambda r: r[0] == "Y" and r[1] == "N"
    if ok(r1) and ok(r2): return "accepted"
    if r1[0] == r2[0] == "N" or (r1[0] == "Y" and r2[0] == "Y" and r1[1] == r2[1] == "Y"): return "rejected"
    return "disagreement"
