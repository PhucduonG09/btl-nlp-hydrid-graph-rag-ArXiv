"""In-memory BM25 for the fixed paper corpus, with query-time parameter control."""
from collections import Counter,defaultdict
import numpy as np
from scripts.run_baselines import tok

class BM25Index:
    def __init__(self,papers):
        self.ids=[p['paper_id'] for p in papers];postings=defaultdict(list);lengths=[]
        for i,p in enumerate(papers):
            tokens=tok(p['title']+'. '+p['abstract']);lengths.append(len(tokens))
            for term,freq in Counter(tokens).items():postings[term].append((i,freq))
        self.lengths=np.asarray(lengths);self.avg=float(self.lengths.mean())
        self.postings={t:np.asarray(v,dtype=np.int32).T for t,v in postings.items()}
        n=len(self.ids)
        self.idf={t:float(np.log(n-len(v)+.5)-np.log(len(v)+.5)) for t,v in postings.items()}
        self.average_idf=sum(self.idf.values())/len(self.idf);self._weights={}
    def scores(self,query,k1=1.5,b=.75,epsilon=.25):
        result=np.zeros(len(self.ids))
        for term in tok(query):
            if term not in self.postings:continue
            key=(term,k1,b,epsilon)
            if key not in self._weights:
                doc,freq=self.postings[term]
                idf=self.idf[term] if self.idf[term]>=0 else epsilon*self.average_idf
                value=idf*freq*(k1+1)/(freq+k1*(1-b+b*self.lengths[doc]/self.avg))
                self._weights[key]=(doc,value)
            doc,value=self._weights[key];result[doc]+=value
        return result
    def rank(self,query,k1=1.5,b=.75,limit=100):
        return [self.ids[i] for i in np.argsort(-self.scores(query,k1,b),kind='stable')[:limit]]
