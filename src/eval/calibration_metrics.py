"""Human-versus-judge agreement; no automatic labels become human ratings."""
from collections import Counter

def agreement_metrics(pairs,labels):
    confusion=Counter(pairs);n=sum(confusion.values())
    if not n:return dict(n=0,agreement=None,cohen_kappa=None,macro_f1=None,confusion=[])
    if any(a not in labels or b not in labels for a,b in confusion):raise ValueError('Unknown calibration label')
    observed=sum(v for (a,b),v in confusion.items() if a==b)/n
    expected=sum(sum(v for (a,b),v in confusion.items() if a==c)*sum(v for (a,b),v in confusion.items() if b==c) for c in labels)/(n*n)
    scores=[]
    for c in labels:
        denominator=sum(v for (a,b),v in confusion.items() if a==c)+sum(v for (a,b),v in confusion.items() if b==c)
        scores.append(2*confusion[(c,c)]/denominator if denominator else 0)
    return dict(n=n,agreement=observed,cohen_kappa=(observed-expected)/(1-expected) if expected<1 else None,
        macro_f1=sum(scores)/len(labels),macro_f1_classes=list(labels),
        confusion=[dict(human=a,judge=b,n=v) for (a,b),v in sorted(confusion.items())])

def judge_support_label(record):
    # Absence/abstention statements are not positive evidence-supported claims.
    if record['quality']=='abstained' or record.get('faithfulness_raw') is None:return 'not_applicable'
    claims=record['claims']
    if not claims:return 'not_applicable'
    supported=sum(c['supported'] for c in claims)
    return 'all_supported' if supported==len(claims) else ('mixed' if supported else 'unsupported')
