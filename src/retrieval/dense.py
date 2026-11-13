import json, os, numpy as np, faiss, torch
from sentence_transformers import SentenceTransformer
from .base import Hit, Retriever

# bge v1.5 tiếng Anh: instruction cho query ngắn -> passage dài. Thử có/không trên dev set.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

class DenseRetriever(Retriever):
    name = "dense"
    def __init__(self, chunks, model_name="BAAI/bge-base-en-v1.5", index_dir=None, use_prefix=True, batch_size=64):
        self.use_prefix = use_prefix
        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        idx_path = index_dir and os.path.join(index_dir, "dense.faiss")
        ids_path = index_dir and os.path.join(index_dir, "dense_ids.json")
        if idx_path and os.path.exists(idx_path):
            self.index = faiss.read_index(idx_path)
            self.ids = json.load(open(ids_path))        # thứ tự id PHẢI khớp thứ tự vector
        else:
            self.ids = [c["chunk_id"] for c in chunks]
            emb = self.model.encode([c["text"] for c in chunks], batch_size=batch_size,
                                    normalize_embeddings=True, show_progress_bar=True).astype("float32")
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
                faiss.write_index(self.index, idx_path)
                json.dump(self.ids, open(ids_path, "w"))

    def retrieve(self, query, k):
        q = (QUERY_PREFIX if self.use_prefix else "") + query
        qv = self.model.encode([q], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, k)
        return [Hit(self.ids[i], float(d), self.name) for d, i in zip(D[0], I[0]) if i != -1]