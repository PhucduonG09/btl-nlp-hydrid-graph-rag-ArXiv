import pytest
from src.retrieval.rerank import reorder_candidates

def test_reranker_preserves_pool_and_respects_depth():
    assert reorder_candidates(['a','b','c'],dict(a=1,b=2,c=99),depth=2)==['b','a','c']

@pytest.mark.parametrize('scores',[dict(a=float('nan'),b=1),dict(a=1)])
def test_reranker_rejects_invalid_scores(scores):
    with pytest.raises(ValueError):reorder_candidates(['a','b'],scores)
