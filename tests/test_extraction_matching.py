from src.eval.extraction_matching import maximum_matching,relation_counts

def rel(name,kind='USES'):return dict(target=name,relation=kind)
def test_duplicate_predictions_cannot_both_score_true():
    assert relation_counts([rel('BERT'),rel('BERT')],[rel('BERT')])==dict(tp=1,fp=1,fn=0)

def test_matching_finds_augmenting_path_instead_of_greedy_assignment():
    assert len(maximum_matching(['flexible','restricted'],['a','b'],lambda p,g:p=='flexible' or g=='a'))==2

def test_relation_type_is_required_for_typed_score():
    assert relation_counts([rel('BERT','PROPOSES')],[rel('BERT')])==dict(tp=0,fp=1,fn=1)
    assert relation_counts([rel('BERT','PROPOSES')],[rel('BERT')],typed=False)==dict(tp=1,fp=0,fn=0)
