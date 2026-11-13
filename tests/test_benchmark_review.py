from copy import deepcopy
import pytest
from src.eval.benchmark_review import save,decisions,effective,content_status,fingerprint,curated_rows

def row():return dict(id='test',type='single',question='Original?',reference_answer='Answer',gold_ids=['paper'],origin='new_authored_draft',review_status='pending')
def test_decision_survives_reload_without_mutating_source(tmp_path):
    source=row();before=deepcopy(source);db=tmp_path/'review.sqlite3'
    save(db,source,question='Edited?',answer='Edited answer',status='approved',reviewer='unit-test')
    result=effective(source,decisions(db))
    assert result['question']=='Edited?' and result['review_status']=='approved'
    assert source==before

def test_changed_draft_invalidates_older_approval(tmp_path):
    source=row();db=tmp_path/'review.sqlite3'
    save(db,source,question='Original?',answer='Answer',status='approved',reviewer='unit-test')
    source['reference_answer']='Different claim'
    assert effective(source,decisions(db))['review_status']=='pending'

def test_approved_bridge_is_read_only(tmp_path):
    source=row();source['origin']='bridge_v1_approved'
    with pytest.raises(ValueError):save(tmp_path/'r.sqlite3',source,question='Changed?',answer='X',status='approved',reviewer='unit-test')

def test_review_requires_identified_reviewer(tmp_path):
    with pytest.raises(ValueError):save(tmp_path/'r.sqlite3',row(),question='Q?',answer='A',status='approved',reviewer='')

def test_changed_source_invalidates_older_approval(tmp_path):
    source=row();source['source_text_sha256']='original';db=tmp_path/'review.sqlite3'
    save(db,source,question='Original?',answer='Answer',status='approved',reviewer='unit-test')
    source['source_text_sha256']='changed'
    assert effective(source,decisions(db))['review_status']=='pending'

def test_source_review_is_invalidated_by_changed_answer():
    source=row();audit={source['id']:dict(status='cleared',source_fingerprint=fingerprint(source))}
    assert content_status(source,audit)=='cleared'
    source['reference_answer']='Another answer'
    assert content_status(source,audit)=='needs_review'

def test_human_rejection_is_not_overruled_by_source_check():
    source=row();audit={source['id']:dict(status='cleared',source_fingerprint=fingerprint(source))}
    source['review_status']='rejected'
    assert content_status(source,audit)=='needs_replacement'

def test_assisted_curation_is_separate_from_user_approval():
    source=row();audit={source['id']:dict(status='cleared',source_fingerprint=fingerprint(source))}
    assert curated_rows([source],audit)==[]
    result=curated_rows([source],audit,allow_assisted=True)
    assert result[0]['curation_provenance']=='assistant_source_review'
    assert result[0]['review_status']=='pending'
    source['review_status']='rejected'
    assert curated_rows([source],audit,allow_assisted=True)==[]
