import numpy as np
from src.retrieval.offline import BM25Index
from src.eval.benchmark_metrics import selection_score

def test_bm25_term_frequency_matches_formula_and_repeated_queries():
    papers=[dict(paper_id='a',title='rare rare',abstract=''),dict(paper_id='b',title='other',abstract=''),dict(paper_id='c',title='third',abstract='')]
    index=BM25Index(papers);scores=index.scores('rare')
    expected=np.log(2.5/1.5)*2*2.5/(2+1.5*(.25+.75*2/(4/3)))
    assert np.isclose(scores[0],expected) and scores[1]==0
    assert np.allclose(index.scores('rare rare'),scores*2)
    assert index.rank('rare')[0]=='a'

def test_selection_balances_question_types_instead_of_sample_counts():
    a={'single':{'n':90,'All@10':1,'MRR@10':1},'bridge':{'n':30,'All@10':0,'MRR@10':0}}
    b={'single':{'n':90,'All@10':.8,'MRR@10':.8},'bridge':{'n':30,'All@10':.8,'MRR@10':.8}}
    assert selection_score(b)>selection_score(a)
