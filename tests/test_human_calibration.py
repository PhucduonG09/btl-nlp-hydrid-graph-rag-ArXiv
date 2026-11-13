import pytest
from src.eval.human_calibration import ratings,save_rating,resolved_ratings,resolved_ratings_v2

def test_human_ratings_need_an_identified_reviewer(tmp_path):
    with pytest.raises(ValueError):save_rating(tmp_path/'r.sqlite3',sample_id='H001',reviewer='',quality='correct',support='all_supported')
def test_abstention_is_not_stored_as_perfect_source_support(tmp_path):
    path=tmp_path/'r.sqlite3'
    with pytest.raises(ValueError):save_rating(path,sample_id='H001',reviewer='test',quality='abstained',support='all_supported')
    save_rating(path,sample_id='H001',reviewer='test',quality='abstained',support='not_applicable')
    assert ratings(path)[0]['support']=='not_applicable'

def test_disagreement_and_reference_issues_remain_pending():
    base=dict(sample_id='H001',quality='correct',support='all_supported',reference_issue=0)
    resolved,pending=resolved_ratings([base,dict(base,quality='partial')],{'H001','H002'})
    assert not resolved and pending=={'H001':'disagreement','H002':'unrated'}
    assert resolved_ratings([dict(base,reference_issue=1)],{'H001'})[1]['H001']=='reference_issue'
    assert resolved_ratings([base,base],{'H001'})[0]['H001']['n_reviewers']==2


def test_v2_reference_issue_is_flag_but_disagreement_still_pending():
    base=dict(sample_id='H001',quality='abstained',support='not_applicable',reference_issue=1)
    resolved,pending=resolved_ratings_v2([base,dict(base,reference_issue=0)],{'H001'})
    assert resolved['H001']['reference_issue'] and not resolved['H001']['reference_issue_all'] and not pending
    resolved,pending=resolved_ratings_v2([base,dict(base,quality='incorrect',support='mixed')],{'H001'})
    assert pending['H001']=='disagreement' and not resolved
    assert resolved_ratings_v2([],{'H001'})[1]['H001']=='unrated'
