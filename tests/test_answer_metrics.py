from src.eval.answer_metrics import paper_clusters,summarize_answers

def test_abstention_is_excluded_from_faithfulness_denominator():
    gen=[dict(refused=True,gold_all_in_context=False,invalid_citations=[])]
    judge=[dict(quality='abstained',relevance=1,claims=[],faithfulness_raw=None,supported_claims_with_invalid_quote=0)]
    result=summarize_answers(gen,judge)
    assert result['correct_rate']==0 and result['faithfulness_macro_nonabstaining'] is None
    assert result['claim_denominator']==0 and result['n']==1
    judge[0].update(faithfulness_raw=1,claims=[dict(supported=True)])
    assert summarize_answers(gen,judge)['faithfulness_macro_nonabstaining'] is None

def test_paper_clusters_include_transitive_overlap():
    qa=[dict(id='a',gold_ids=['1','2']),dict(id='b',gold_ids=['2','3']),dict(id='c',gold_ids=['3']),dict(id='d',gold_ids=['4'])]
    assert sorted(map(len,paper_clusters(qa)))==[1,3]
