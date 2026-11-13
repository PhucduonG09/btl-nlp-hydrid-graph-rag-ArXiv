"""Paper-component split preserving previously locked bridge assignments."""
import random

def split_rows(rows,fixed,seed=42,new_test_fraction=.3):
    parent={r['id']:r['id'] for r in rows}
    def root(qid):
        while parent[qid]!=qid:
            parent[qid]=parent[parent[qid]];qid=parent[qid]
        return qid
    seen={}
    for row in rows:
        for paper in row['gold_ids']:
            if paper in seen:parent[root(row['id'])]=root(seen[paper])
            else:seen[paper]=row['id']
    groups={}
    for row in rows:groups.setdefault(root(row['id']),[]).append(row)
    assigned={};fresh=[]
    for group,items in groups.items():
        labels={fixed[r['id']] for r in items if r['id'] in fixed}
        if len(labels)>1:raise ValueError('A paper component crosses the locked bridge split')
        if labels:assigned[group]=next(iter(labels))
        elif any(r.get('origin')=='new_authored_draft' for r in items):fresh.append(group)
        else:assigned[group]='dev'  # Historically exposed legacy questions stay in DEV unless anchored.
    fresh.sort();random.Random(seed).shuffle(fresh)
    n_test=round(len(fresh)*new_test_fraction)
    for i,group in enumerate(fresh):assigned[group]='test' if i<n_test else 'dev'
    result={part:[r for r in rows if assigned[root(r['id'])]==part] for part in ['dev','test']}
    assert {p for r in result['dev'] for p in r['gold_ids']}.isdisjoint(p for r in result['test'] for p in r['gold_ids'])
    return result
