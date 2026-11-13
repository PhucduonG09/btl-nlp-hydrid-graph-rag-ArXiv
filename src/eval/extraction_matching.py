"""Maximum-cardinality, one-to-one matching of relation instances."""
from src.kg.aliases import match

def maximum_matching(predictions,gold,compatible):
    adjacency=[[j for j,g in enumerate(gold) if compatible(p,g)] for p in predictions]
    owner={}
    def augment(i,seen):
        for j in adjacency[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owner or augment(owner[j],seen):
                owner[j]=i;return True
        return False
    for i in range(len(predictions)):augment(i,set())
    return sorted((i,j) for j,i in owner.items())

def relation_counts(predictions,gold,typed=True):
    def compatible(p,g):
        return (not typed or p['relation']==g['relation']) and match(p['target'],g['target'],loose=True)
    pairs=maximum_matching(predictions,gold,compatible)
    return dict(tp=len(pairs),fp=len(predictions)-len(pairs),fn=len(gold)-len(pairs))

def prf(counts):
    tp,fp,fn=(counts[k] for k in ['tp','fp','fn'])
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return dict(**counts,P=p,R=r,F1=2*p*r/(p+r) if p+r else 0.)
