"""Thống kê retrieval TEST đã khóa (offline, không API): McNemar chính xác, CI 95% bootstrap theo thành phần gold,
Clopper-Pearson, và bảng lỗi của câu Graph hỏng. Chỉ chạy sau execute (test đã mở theo protocol).
  python -X utf8 scripts/analyze_test_retrieval_v2.py
"""
import json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/benchmark_v2_assisted/test_01'
METHODS=['bm25','dense','rrf','text_expansion','graph'];SEED=20261112;B=10000
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
assert (OUT/'execution_complete.json').exists(),'Test chưa được execute'
qa={q['id']:q for q in load(ROOT/'data/eval/benchmark_v2_assisted/qa_test.jsonl')}
ret={r['id']:r for r in load(OUT/'retrieval.jsonl')}
def comp(qa):
    parent={}
    def f(x):
        while parent.setdefault(x,x)!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for q in qa.values():
        g=q['gold_ids']
        for a in g[1:]:parent[f(a)]=f(g[0])
        f(g[0])
    return {i:f(q['gold_ids'][0]) for i,q in qa.items()}
C=comp(qa)
def hit(i,m,k=10):return set(qa[i]['gold_ids'])<=set(ret[i]['rankings'][m][:k])
def rank(i,m,pid):
    r=ret[i]['rankings'][m];return r.index(pid)+1 if pid in r else None
def binom_cdf(k,n,p=.5):return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
def mcnemar(b,c):
    n=b+c
    return 1.0 if n==0 else min(1.0,2*binom_cdf(min(b,c),n))
def cp(k,n,a=.05):
    def bs(f,lo,hi):
        for _ in range(80):
            mid=(lo+hi)/2
            lo,hi=(mid,hi) if f(mid) else (lo,mid)
        return (lo+hi)/2
    lo=0.0 if k==0 else bs(lambda p:1-binom_cdf(k-1,n,p)<a/2,0,1)
    hi=1.0 if k==n else bs(lambda p:binom_cdf(k,n,p)>a/2,0,1)
    return lo,hi
def cluster_ci(ids,fn):
    rng=np.random.default_rng(SEED);by=defaultdict(list)
    for i in ids:by[C[i]].append(i)
    keys=list(by);vals=[]
    for _ in range(B):
        pick=[x for k in rng.integers(0,len(keys),len(keys)) for x in by[keys[k]]]
        vals.append(fn(pick))
    return [round(float(np.percentile(vals,2.5)),3),round(float(np.percentile(vals,97.5)),3)]
res=dict(note='Thống kê retrieval TEST (28 câu). Bootstrap theo thành phần gold. Graph đạt 100% reachability trên test nên so sánh Graph–RRF phản ánh cách dựng bridge.',
         n_components=len(set(C.values())),groups={})
for name,ids in [('bridge',[i for i,q in qa.items() if q['type']=='bridge']),('single',[i for i,q in qa.items() if q['type']=='single'])]:
    g={}
    for m in METHODS:
        k=sum(hit(i,m) for i in ids);g[m]=dict(all_at_10=f'{k}/{len(ids)}',rate=round(k/len(ids),3),clopper_pearson_95=[round(x,3) for x in cp(k,len(ids))],
            cluster_bootstrap_95=cluster_ci(ids,lambda s,m=m:float(np.mean([hit(i,m) for i in s]))))
    for other in ('rrf','bm25','dense','text_expansion'):
        b=sum(hit(i,'graph') and not hit(i,other) for i in ids);c=sum(hit(i,other) and not hit(i,'graph') for i in ids)
        g[f'graph_vs_{other}']=dict(graph_only=b,other_only=c,mcnemar_exact_p=mcnemar(b,c),
            diff_cluster_bootstrap_95=cluster_ci(ids,lambda s,o=other:float(np.mean([hit(i,'graph') for i in s])-np.mean([hit(i,o) for i in s]))))
    res['groups'][name]=g
fails=[]
for i,q in qa.items():
    if q['type']=='bridge' and not hit(i,'graph'):
        g=q['gold_ids'];r=ret[i]
        fails.append(dict(id=i,gold=g,ranks={m:{p:rank(i,m,p) for p in g} for m in METHODS},seeds=r.get('seeds'),
            graph_paths_to_gold={p:str(r.get('paths',{}).get(p))[:200] for p in g} if isinstance(r.get('paths'),dict) else str(r.get('paths'))[:400]))
res['graph_failures_on_bridge']=fails
(OUT/'retrieval_stats.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=1)[:6000])
