from pathlib import Path
from streamlit.testing.v1 import AppTest
import pytest

@pytest.fixture(autouse=True)
def isolated_ratings(monkeypatch,tmp_path):
    # Hill can be actively rating while tests run: never inspect/compare the
    # changing production database, and never allow a test to write to it.
    import src.eval.human_calibration as human
    original=human.ratings;db=tmp_path/'ratings.sqlite3'
    monkeypatch.setattr(human,'ratings',lambda ignored:original(db))
    def no_save(*args,**kwargs):raise AssertionError('Read-only UI test attempted a rating write')
    monkeypatch.setattr(human,'save_rating',no_save)
    return db

def test_blind_answer_pack_loads_without_writing_ratings(isolated_ratings):
    root=Path(__file__).resolve().parents[1]
    db=isolated_ratings
    before=db.read_bytes() if db.exists() else None
    app=AppTest.from_file(str(root/'review_answers.py')).run(timeout=40)
    assert not app.exception and len(app.selectbox(key='human_sample').options)==50
    assert app.text_input(key='human_reviewer').value==''
    app.selectbox(key='human_sample').select('H050').run(timeout=40)
    assert not app.exception and len(app.form)==1
    assert (db.read_bytes() if db.exists() else None)==before

def test_assistant_view_is_read_only_and_keeps_human_labels_separate(isolated_ratings):
    root=Path(__file__).resolve().parents[1]
    db=isolated_ratings
    before=db.read_bytes() if db.exists() else None
    app=AppTest.from_file(str(root/'review_answers.py'))
    app.query_params['view']='assistant';app.run(timeout=40)
    assert not app.exception and len(app.form)==0
    assert app.metric[0].value=='33' and app.metric[3].value=='13'
    assert app.selectbox(key='assistant_review_sample').options==['H005','H020','H036','H042','H048']
    app.selectbox(key='assistant_review_sample').select('H036').run(timeout=40)
    assert not app.exception and len(app.warning)==1
    app.selectbox(key='assistant_review_group').select('Tất cả 50 mẫu').run(timeout=40)
    assert len(app.selectbox(key='assistant_review_sample').options)==50
    assert (db.read_bytes() if db.exists() else None)==before

def test_shared_review_never_reveals_assistant_suggestions(monkeypatch):
    monkeypatch.setenv('REVIEW_SHARED_MODE','1')
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'review_answers.py'))
    app.query_params['view']='assistant';app.run(timeout=40)
    assert not app.exception and len(app.form)==1
    assert app.text_input(key='human_reviewer').value==''
    assert not any('assistant_review' in str(e.key) for e in app.selectbox)
    assert app.title[0].value=='Chấm đáp án theo nguồn'
