from collections import defaultdict
from .base import Hit, Retriever

class RRFRetriever(Retriever):
    def __init__(self, retrievers, k_rrf=60, depth=50, weights=None, name="rrf"):
        self.retrievers, self.k_rrf, self.depth, self.name = retrievers, k_rrf, depth, name
        self.weights = weights or [1.0] * len(retrievers)

    def retrieve(self, query, k):
        scores = defaultdict(float)
        for w, r in zip(self.weights, self.retrievers):
            for rank, h in enumerate(r.retrieve(query, self.depth), start=1):
                scores[h.chunk_id] += w / (self.k_rrf + rank)
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [Hit(cid, s, self.name) for cid, s in top]