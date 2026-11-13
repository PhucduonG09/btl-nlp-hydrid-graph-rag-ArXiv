from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_review_ui_shows_sources_revisions_and_read_only_bridges():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'review_benchmark.py')).run(timeout=40)
    assert not app.exception
    assert app.selectbox(key='review_group').value=='Các câu đã thay'
    assert set(app.selectbox(key='review_id').options)=={'q056','s2006','s2019'}
    assert app.metric[1].value=='150'
    assert app.metric[2].value=='0'
    for qid in ['s2006','s2019','q056']:
        app.selectbox(key='review_id').select(qid).run(timeout=40)
        assert not app.exception
        assert len(app.text_area)>=2
    app.selectbox(key='review_group').select('Nháp mới').run(timeout=40)
    app.selectbox(key='review_id').select('s2001').run(timeout=40)
    assert app.selectbox(key='review_id').value=='s2001'
    assert len(app.form)==1
    app.selectbox(key='review_group').select('Câu cũ có đề xuất sửa').run(timeout=40)
    app.selectbox(key='review_id').select('q033').run(timeout=40)
    assert not app.exception and len(app.info)==1
    app.selectbox(key='review_group').select('Toàn bộ câu single cũ').run(timeout=40)
    assert not app.exception
    assert not app.checkbox(key='legacy_attested').value
    app.selectbox(key='review_group').select('Bridge đã duyệt').run(timeout=40)
    assert not app.exception
    assert len(app.form)==0
