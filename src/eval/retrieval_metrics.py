import math

def dedup_papers(hits, store):
    seen, out = set(), []
    for h in hits:
        pid = store[h.chunk_id]["paper_id"]
        if pid not in seen:
            seen.add(pid); out.append(pid)
    return out

def recall_at_k(ranked, gold, k):  return len(set(ranked[:k]) & gold) / len(gold)
def all_evidence_at_k(ranked, gold, k):  return float(gold <= set(ranked[:k]))

def mrr(ranked, gold):
    return next((1 / i for i, d in enumerate(ranked, 1) if d in gold), 0.0)

def ndcg_at_k(ranked, gold, k):
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in gold)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg