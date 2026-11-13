from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_dev_explorer_loads_and_switches_method():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=40)
    assert not app.exception
    assert app.metric[0].value=='20,500'
    app.toggle(key='show_gold').set_value(True).run(timeout=40)
    assert not app.exception and any('gold_id:' in e.value for e in app.text)
    app.selectbox(key='question_id').select('bn001').run(timeout=40)
    assert not app.exception and any('hop2_id:' in e.value for e in app.text)
    app.selectbox(key='dataset').select('Bridge v1 · DEV').run(timeout=40)
    app.selectbox(key='method').select('linked_k1_rrf').run(timeout=40)
    assert not app.exception
    assert len(app.metric)==3
    app.toggle(key='show_gold').set_value(True).run(timeout=40)
    assert not app.exception
    assert any('hop2_id:' in element.value for element in app.text)

def test_live_mode_does_not_call_model_on_open_or_empty_submit():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=40)
    app.radio(key='mode').set_value('Hỏi đáp').run(timeout=40)
    assert not app.exception and len(app.form)==1
    app.button(key='live_submit').click().run(timeout=40)
    assert not app.exception and any('10 ký tự' in e.value for e in app.error)
