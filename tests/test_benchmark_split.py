import pytest
from src.eval.benchmark_split import split_rows

def q(qid,papers,origin='legacy_single'):return dict(id=qid,gold_ids=papers,origin=origin)
def test_legacy_sharing_a_locked_test_paper_follows_that_component():
    rows=[q('bridge',['A','B']),q('old',['A']),q('other',['C'])]
    result=split_rows(rows,{'bridge':'test'})
    assert {r['id'] for r in result['test']}=={'bridge','old'}
    assert result['dev'][0]['id']=='other'

def test_conflicting_locked_labels_are_rejected():
    with pytest.raises(ValueError):split_rows([q('a',['P']),q('b',['P'])],{'a':'dev','b':'test'})

def test_fresh_paper_split_is_reproducible_and_disjoint():
    rows=[q(f'n{i:02}',[str(i)],'new_authored_draft') for i in range(10)]
    a=split_rows(rows,{});b=split_rows(rows,{})
    assert a==b and len(a['test'])==3
    assert {r['id'] for r in a['dev']}.isdisjoint(r['id'] for r in a['test'])
