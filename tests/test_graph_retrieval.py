from src.kg.mentions import build_mention_graph,MentionIndex
from src.retrieval.graph import GraphSearch

def paper(pid,text):return {'paper_id':pid,'title':text,'abstract':''}

def test_graph_expands_only_through_real_mentions_and_explains_path():
    ps=[paper('A','NewAgent uses NamedBench'),paper('B','NamedBench data collection'),paper('C','UnrelatedSystem')]
    graph=build_mention_graph(ps);search=GraphSearch(MentionIndex(graph))
    ranking,paths=search.expand(['A'])
    assert 'B' in ranking and 'C' not in ranking
    assert paths['B'][0]['A']=='A' and paths['B'][0]['B']=='B'
    assert paths['B'][0]['relation']=='MENTIONS'
    ppr,_=search.ppr(['A'])
    assert 'B' in ppr and 'C' not in ppr

def test_entity_graph_does_not_link_different_model_versions():
    graph=build_mention_graph([paper('A','GPT-4 is evaluated'),paper('B','GPT-4V is evaluated')])
    ranking,_=GraphSearch(MentionIndex(graph)).expand(['A'])
    assert 'B' not in ranking
