import re

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def triples(rels):
    return {(r["relation"], norm(r["target"])) for r in rels}

def micro_prf(pred_by_paper: dict, gold_by_paper: dict):
    """Micro P/R/F1 trên tập (relation, target) của từng paper. Khớp sau khi chuẩn hóa chuỗi."""
    tp = fp = fn = 0
    for pid, gold in gold_by_paper.items():
        g, p = triples(gold), triples(pred_by_paper.get(pid, []))
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"P": prec, "R": rec, "F1": f1, "tp": tp, "fp": fp, "fn": fn}