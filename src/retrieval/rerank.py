"""Reorder a bounded candidate pool without adding documents outside that pool."""
import math

def reorder_candidates(ranking,scores,depth=50):
    if len(ranking)!=len(set(ranking)):raise ValueError('Duplicate candidates')
    head=ranking[:depth]
    if any(pid not in scores or not math.isfinite(float(scores[pid])) for pid in head):
        raise ValueError('Missing or non-finite candidate scores')
    position={pid:i for i,pid in enumerate(head)}
    ordered=sorted(head,key=lambda pid:(-float(scores[pid]),position[pid]))
    return ordered+ranking[depth:]
