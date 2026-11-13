"""Bounded paper-to-paper expansion and implicit bipartite personalized PageRank."""
import math
from collections import defaultdict
import numpy as np

class GraphSearch:
    def __init__(self,index):
        self.index=index;self.ids=index.data['paper_ids'];self.lookup={p:i for i,p in enumerate(self.ids)}
        self.entities=list(index.entity_papers);emap={e:i for i,e in enumerate(self.entities)}
        pidx=[];eidx=[];weights=[]
        for pid,entities in index.paper_entities.items():
            for e in entities:
                pidx.append(self.lookup[pid]);eidx.append(emap[e]);weights.append(index.idf(e))
        self.pi=np.asarray(pidx,dtype=np.int32);self.ei=np.asarray(eidx,dtype=np.int32);weights=np.asarray(weights)
        psum=np.bincount(self.pi,weights=weights,minlength=len(self.ids));esum=np.bincount(self.ei,weights=weights,minlength=len(self.entities))
        self.pw=weights/psum[self.pi];self.ew=weights/esum[self.ei];self.dangling=psum==0
    def expand(self,seeds,hops=1,beam=100,entity_limit=20):
        # One hop is one Paper -> Entity <- Paper transition (two incidence edges).
        scores={pid:1/(rank+1) for rank,pid in enumerate(seeds)}
        frontier=dict(scores);paths={pid:[] for pid in seeds}
        for depth in range(hops):
            next_scores=defaultdict(float);next_paths={}
            for pid,weight in frontier.items():
                shared=sorted((e for e in self.index.paper_entities.get(pid,[]) if len(self.index.entity_papers[e])>=2),key=lambda e:(-self.index.idf(e),e))[:entity_limit]
                for e in shared:
                    neighbors=self.index.entity_papers[e]
                    for other in neighbors:
                        if other==pid:continue
                        value=weight*self.index.idf(e)/(math.log2(len(neighbors)+1)*10)*(1.5 if e in self.index.title_entities.get(other,set()) else 1)
                        next_scores[other]+=value
                        if other not in next_paths or value>next_paths[other][0]:
                            next_paths[other]=(value,paths.get(pid,[])+[dict(A=pid,entity=e,B=other,relation='MENTIONS')])
            frontier=dict(sorted(next_scores.items(),key=lambda x:(-x[1],x[0]))[:beam])
            for pid,score in frontier.items():
                if score>scores.get(pid,0):scores[pid]=score;paths[pid]=next_paths[pid][1]
        ranked=[p for p,_ in sorted(scores.items(),key=lambda x:(-x[1],x[0]))]
        return ranked,paths
    def ppr(self,seeds,alpha=.85,iterations=60,tolerance=1e-8):
        restart=np.zeros(len(self.ids))
        for rank,pid in enumerate(seeds):
            if pid in self.lookup:restart[self.lookup[pid]]=1/(rank+1)
        if not restart.sum():return [],{}
        restart/=restart.sum();x=restart.copy()
        for _ in range(iterations):
            entity_mass=np.bincount(self.ei,weights=x[self.pi]*self.pw,minlength=len(self.entities))
            moved=np.bincount(self.pi,weights=entity_mass[self.ei]*self.ew,minlength=len(self.ids))
            moved+=x[self.dangling].sum()*restart
            y=(1-alpha)*restart+alpha*moved
            if np.abs(y-x).sum()<tolerance:x=y;break
            x=y
        ranked=[self.ids[i] for i in np.argsort(-x,kind='stable') if x[i]>0]
        # Explain short paths separately; PPR score alone is not a unique path.
        _,paths=self.expand(seeds,hops=2,beam=150)
        return ranked,paths

def fuse_rrf(lists,k=60,weights=None,limit=100):
    scores=defaultdict(float);weights=weights or [1]*len(lists)
    for w,ranking in zip(weights,lists):
        for r,pid in enumerate(ranking[:100],1):scores[pid]+=w/(k+r)
    return [p for p,_ in sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:limit]]

def fuse_weighted(lists,weights=None,limit=100):
    # Rank-normalized scores avoid mixing incomparable BM25/cosine/PPR scales.
    weights=weights or [1/len(lists)]*len(lists);score=defaultdict(float)
    for weight,ranking in zip(weights,lists):
        for rank,pid in enumerate(ranking[:100],1):score[pid]+=weight/rank
    return [p for p,_ in sorted(score.items(),key=lambda x:(-x[1],x[0]))[:limit]]
