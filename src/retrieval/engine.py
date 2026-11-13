"""Fixed DEV-selected retrieval for arbitrary questions, without QA/gold access."""
import hashlib,json,os,threading,time
from pathlib import Path
import numpy as np
from src.retrieval.offline import BM25Index
from src.retrieval.graph import GraphSearch,fuse_rrf
from src.kg.mentions import MentionIndex
from scripts.run_baselines import PREFIX
ROOT=Path(__file__).resolve().parents[2]

class RetrievalEngine:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(4,os.cpu_count() or 1));self.lock=threading.Lock()
        load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        manifest=json.loads((ROOT/'data/eval/benchmark_v2_assisted/manifest.json').read_text(encoding='utf-8'))
        corpus=ROOT/manifest['corpus'];assert hashlib.sha256(corpus.read_bytes()).hexdigest()==manifest['corpus_sha256']
        papers=load(corpus);self.papers={p['paper_id']:p for p in papers};self.ids=list(self.papers);self.bm25=BM25Index(papers)
        graph=json.loads((ROOT/'data/kg/reference_mentions_v2_assisted.json').read_text(encoding='utf-8'))
        assert graph['corpus_sha256']==manifest['corpus_sha256']
        self.mentions=MentionIndex(graph);self.graph=GraphSearch(self.mentions)
        source=load(ROOT/'data/processed/pilot_papers.jsonl')+load(ROOT/'data/processed/distractors_20000.jsonl')
        model='BAAI/bge-base-en-v1.5';source_ids=[p['paper_id'] for p in source]
        key=hashlib.sha1(('|'.join(source_ids)+model).encode()).hexdigest()[:10]
        cached=np.load(ROOT/'data/index'/f'emb_scale_{key}.npy',mmap_mode='r')
        lookup={pid:i for i,pid in enumerate(source_ids)}
        self.embeddings=np.asarray(cached[[lookup[pid] for pid in self.ids]],dtype=np.float32)
        revision='a5beb1e3e68b9ab74eb54cfd186867f64f240e1a'
        roots=[ROOT/'data/cache/huggingface',Path(os.environ.get('HF_HOME','D:/Dev/cache/huggingface'))/'hub']
        snapshot=next((p/'models--BAAI--bge-base-en-v1.5'/'snapshots'/revision for p in roots if (p/'models--BAAI--bge-base-en-v1.5'/'snapshots'/revision).exists()),None)
        if snapshot is None:raise FileNotFoundError('Pinned BGE snapshot is missing from local cache')
        self.encoder=SentenceTransformer(str(snapshot),device='cpu',local_files_only=True,trust_remote_code=False)
    def retrieve(self,question,method='graph'):
        start=time.perf_counter();bm=self.bm25.rank(question,k1=1.5,b=.75)
        with self.lock:vector=self.encoder.encode([PREFIX+question],normalize_embeddings=True,show_progress_bar=False)[0]
        scores=self.embeddings@vector;dense=[self.ids[i] for i in np.argsort(-scores,kind='stable')[:100]]
        base=fuse_rrf([bm,dense],k=10);paths={};seeds=[]
        if method in {'graph','all'}:
            seeds=self.mentions.linked_seeds(question,base,count=3);expanded,paths=self.graph.expand(seeds,hops=1)
            ranking=fuse_rrf([base,expanded],k=60,weights=[1,2])
        elif method=='rrf':ranking=base
        else:raise ValueError('Unknown retrieval method')
        if method=='all':
            expansions=[self.bm25.rank(question+' '+self.papers[pid]['title']+'. '+self.papers[pid]['abstract']) for pid in seeds]
            return dict(rankings=dict(bm25=bm,dense=dense,rrf=base,graph=ranking,
                text_expansion=fuse_rrf([base]+expansions)),paths=paths,seeds=seeds,
                method='fixed_five_methods',latency_seconds=time.perf_counter()-start)
        return dict(ranked=ranking,paths=paths,seeds=seeds,method=method,latency_seconds=time.perf_counter()-start)

def load_engine():return RetrievalEngine()
