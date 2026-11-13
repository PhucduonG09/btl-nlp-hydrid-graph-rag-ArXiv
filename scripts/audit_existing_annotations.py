"""Read-only agreement diagnostic; does not certify independent human authorship."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.iaa import read
from src.kg.aliases import match
from src.eval.extraction_matching import maximum_matching

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kappa(x,y):
    if not x:return None
    po=sum(a==b for a,b in zip(x,y))/len(x)
    pe=sum(x.count(c)*y.count(c) for c in set(x)|set(y))/len(x)**2
    return (po-pe)/(1-pe) if pe<1 else None
def main():
    base=ROOT/'data/eval/annotation';paths=[base/'annotator_A.xlsx',base/'annotator_B.xlsx']
    hashes={p.name:sha(p) for p in paths};a,b=map(read,paths)
    manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
    overlap=[pid for pid in manifest['overlap'] if pid in a and pid in b]
    pairs_total=n_a=n_b=0;ra=[];rb=[];na=[];nb=[];disagreements=[]
    for pid in overlap:
        x=[r for r in a[pid] if r['relation']!='NONE'];y=[r for r in b[pid] if r['relation']!='NONE']
        pairs=maximum_matching(x,y,lambda p,g:match(p['target'],g['target'],loose=True))
        n_a+=len(x);n_b+=len(y);pairs_total+=len(pairs)
        na.append(any(r['relation']=='NONE' for r in a[pid]));nb.append(any(r['relation']=='NONE' for r in b[pid]))
        for i,j in pairs:
            ra.append(x[i]['relation']);rb.append(y[j]['relation'])
            if x[i]['relation']!=y[j]['relation']:
                disagreements.append(dict(paper_id=pid,kind='RELATION',A=x[i],B=y[j]))
        used_a={i for i,_ in pairs};used_b={j for _,j in pairs}
        disagreements.extend(dict(paper_id=pid,kind='ONLY_A',A=r) for i,r in enumerate(x) if i not in used_a)
        disagreements.extend(dict(paper_id=pid,kind='ONLY_B',B=r) for j,r in enumerate(y) if j not in used_b)
        if na[-1]!=nb[-1]:disagreements.append(dict(paper_id=pid,kind='NONE',A=na[-1],B=nb[-1]))
    def safe(v):
        if isinstance(v,set):return sorted(v)
        raise TypeError(type(v).__name__)
    report=dict(source_sha256=hashes,overlap_expected=len(manifest['overlap']),overlap_completed=len(overlap),
                n_A=n_a,n_B=n_b,matched_names=pairs_total,name_instance_F1=2*pairs_total/(n_a+n_b) if n_a+n_b else 0,
                relation_agreement_on_matched=sum(x==y for x,y in zip(ra,rb))/len(ra) if ra else None,
                relation_kappa_on_matched=kappa(ra,rb),NONE_kappa=kappa(na,nb),disagreements=disagreements,
                protocol='One-to-one maximum name matching, no evidence overlap fallback. Relation kappa conditional on matched names; unmatched instances reported separately.',
                provenance='Existing A/B worksheets. Independence, annotator identities and pre-adjudication status require confirmation; this script does not establish them.')
    assert all(sha(p)==hashes[p.name] for p in paths)
    out=ROOT/'results/analysis/annotation_agreement_audit_v1.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=safe),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='disagreements'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
