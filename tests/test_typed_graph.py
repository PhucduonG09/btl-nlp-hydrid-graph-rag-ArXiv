from src.kg.mentions import MentionIndex
from src.kg.typed_reference import TypedGraphSearch

def test_strict_graph_does_not_treat_a_shared_baseline_as_a_proposal():
    data=dict(paper_ids=['a','b','c'],paper_entities={'a':['bench'],'b':['bench'],'c':['bench']},entity_papers={'bench':['a','b','c']},
              title_entities={'a':[],'b':[],'c':[]},alias_expansions={},edge_relations={'a':{'bench':['EVALUATES_ON']},'b':{'bench':['PROPOSES']},'c':{'bench':['USES']}})
    search=TypedGraphSearch(MentionIndex(data));ranked,paths=search.strict_onehop(['a'])
    assert 'b' in ranked and 'c' not in ranked
    assert paths['b'][0]['B_relations']==['PROPOSES']
