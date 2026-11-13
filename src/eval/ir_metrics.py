"""Metric truy xuất với nhiều gold (single-hop có 1 gold, multi-hop có >=2). Quan hệ nhị phân."""
import math


def recall_at_k(ranked, gold, k):
    g = set(gold)
    return len(g & set(ranked[:k])) / len(g)


def hit_all_at_k(ranked, gold, k):
    return float(set(gold) <= set(ranked[:k]))


def mrr_at_k(ranked, gold, k=10):
    g = set(gold)
    for r, d in enumerate(ranked[:k], 1):
        if d in g:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked, gold, k=10):
    g = set(gold)
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(ranked[:k], 1) if d in g)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(g), k) + 1))
    return dcg / idcg if idcg else 0.0


METRICS = {
    "R@1": lambda r, g: recall_at_k(r, g, 1),
    "R@5": lambda r, g: recall_at_k(r, g, 5),
    "R@10": lambda r, g: recall_at_k(r, g, 10),
    "MRR@10": lambda r, g: mrr_at_k(r, g, 10),
    "nDCG@10": lambda r, g: ndcg_at_k(r, g, 10),
    "All@10": lambda r, g: hit_all_at_k(r, g, 10),
}
