"""Agreement Phúc vs duyanh. Chỉ đọc, không ghi DB, không gọi API.
Mặc định so NHÃN GỐC (ratings.before-adjudication.sqlite3) và NHÃN SAU CHỐT (ratings.sqlite3).
  python -X utf8 scripts/inter_rater_agreement_v2.py
"""
import json,sqlite3,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PACK=ROOT/'data/eval/generation_calibration_v2'
def rows(p):
    c=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
    d={}
    for s,r,q,su,ri in c.execute('select sample_id,reviewer,quality,support,reference_issue from ratings'):d.setdefault(s,{})[r]=(q,su,int(ri))
    return d
def kappa(pairs):
    n=len(pairs);po=sum(a==b for a,b in pairs)/n
    ca=Counter(a for a,_ in pairs);cb=Counter(b for _,b in pairs)
    pe=sum(ca[k]*cb[k] for k in set(ca)|set(cb))/(n*n)
    return po,(po-pe)/(1-pe) if pe<1 else None
def report(name,d):
    a,b='Phúc','duyanh';ids=sorted(s for s in d if a in d[s] and b in d[s])
    out=dict(version=name,n=len(ids))
    for label,idx in (('quality',0),('support',1),('reference_issue',2)):
        pairs=[(d[s][a][idx],d[s][b][idx]) for s in ids];po,k=kappa(pairs)
        out[label]=dict(agreement=round(po,4),cohen_kappa=None if k is None else round(k,4),n_disagree=sum(x!=y for x,y in pairs))
    pairs=[(d[s][a][:2],d[s][b][:2]) for s in ids];out['quality_and_support_both']=dict(agreement=round(sum(x==y for x,y in pairs)/len(pairs),4))
    out['disagree_samples']=[s for s in ids if d[s][a][:2]!=d[s][b][:2]]
    return out
res=[report('goc_truoc_thao_luan',rows(PACK/'ratings.before-adjudication.sqlite3')),report('sau_chot',rows(PACK/'ratings.sqlite3'))]
(PACK/'inter_rater_agreement.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
