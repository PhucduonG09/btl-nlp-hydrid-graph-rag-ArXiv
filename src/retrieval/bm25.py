import re, numpy as np
from rank_bm25 import BM25Okapi
from .base import Hit, Retriever

STOP = set("a an the of and or in on for to with by from is are was were be this that we our these which as at it its".split())

def tok(s: str):
    # giữ dấu gạch nối trong từ: 'retrieval-augmented', 'gpt-4'
    return [w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", s.lower()) if w not in STOP]

class BM25Retriever(Retriever):
    name = "bm25"
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.ids = [c["chunk_id"] for c in chunks]
        self.bm25 = BM25Okapi([tok(c["text"]) for c in chunks], k1=k1, b=b)

    def retrieve(self, query, k):
        s = self.bm25.get_scores(tok(query))
        top = np.argsort(-s)[:k]
        return [Hit(self.ids[i], float(s[i]), self.name) for i in top]