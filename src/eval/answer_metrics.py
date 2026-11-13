"""Descriptive answer metrics. Citation validity does not imply truth."""
from collections import Counter
import numpy as np

def latest_records(rows):return {(r['id'],r['system']):r for r in rows}

def summarize_answers(generated,judged):
    assert generated and len(generated)==len(judged)
    assert all(not r.get('error') for r in generated+judged)
    factual=[r for r in judged if r['quality']!='abstained' and r.get('faithfulness_raw') is not None]
    claims=[c for r in factual for c in r['claims']]
    faithful=[r['faithfulness_raw'] for r in factual]
    quality=Counter(r['quality'] for r in judged);n=len(generated)
    return dict(n=n,quality_counts=dict(quality),correct_rate=quality['correct']/n,
        program_refusal_count=sum(r['refused'] for r in generated),
        all_gold_in_packed_context=sum(r['gold_all_in_context'] for r in generated),
        answers_with_invalid_citations=sum(bool(r['invalid_citations']) for r in generated),
        mean_relevance=float(np.mean([r['relevance'] for r in judged])),
        faithfulness_macro_nonabstaining=float(np.mean(faithful)) if faithful else None,
        faithfulness_micro_nonabstaining=sum(c['supported'] for c in claims)/len(claims) if claims else None,
        faithfulness_answer_denominator=len(faithful),claim_denominator=len(claims),
        supported_claims_with_invalid_quote=sum(r['supported_claims_with_invalid_quote'] for r in judged),
        human_calibrated=False)

def paper_clusters(qa):
    """Connect queries sharing any gold paper, including transitive overlap."""
    parent={q['id']:q['id'] for q in qa};owner={}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    for q in qa:
        for paper in q['gold_ids']:
            if paper in owner:parent[root(q['id'])]=root(owner[paper])
            else:owner[paper]=q['id']
    groups={}
    for q in qa:groups.setdefault(root(q['id']),[]).append(q['id'])
    return list(groups.values())

def paired_correctness_ci(qa,judges,replicates=2000,seed=20261109):
    groups=paper_clusters(qa);rng=np.random.default_rng(seed)
    delta={q['id']:int(judges[(q['id'],'graph')]['quality']=='correct')-int(judges[(q['id'],'rrf')]['quality']=='correct') for q in qa}
    sums=np.array([sum(delta[i] for i in g) for g in groups]);sizes=np.array([len(g) for g in groups])
    draws=rng.integers(0,len(groups),size=(replicates,len(groups)))
    samples=sums[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    return dict(delta_graph_minus_rrf=float(np.mean(list(delta.values()))),
        cluster_bootstrap_95_percentile=np.percentile(samples,[2.5,97.5]).tolist(),
        paper_connected_clusters=len(groups),replicates=replicates,seed=seed,
        interpretation='Exploratory DEV interval; configurations selected on this DEV set, uncalibrated automatic judge. Not confirmatory test evidence.')
