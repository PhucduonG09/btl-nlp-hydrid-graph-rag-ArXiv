from src.kg.evidence import locate_evidence
from src.kg.aliases import match,merge_acronyms

def test_latex_quote_returns_original_span():
    source=r'We name it \emph{Memento}, with $87.88\%$ Pass@$3$.'
    result=locate_evidence(source,'Memento, with 87.88% Pass@3.')
    assert result is not None and result.match_kind=='format_normalized'
    assert result.text==source[result.start:result.end]
    assert r'\emph' not in result.text or 'Memento' in result.text

def test_evidence_does_not_accept_changed_numbers_or_negation():
    source='We do not use GPT-4; accuracy is 34%.'
    assert locate_evidence(source,'We use GPT-4') is None
    assert locate_evidence(source,'accuracy is 39%') is None
    assert locate_evidence(source,'we use a new method') is None

def test_names_do_not_merge_versions_or_word_substrings():
    for a,b in [('ToM','automatic evaluation'),('GPT-4','GPT-4V'),('HelpSteer','HelpSteer2'),('SugarCrepe','SugarCrepe++'),('BERT','RoBERTa')]:
        assert not match(a,b,loose=True),(a,b)
    assert match('BeaverTails','BeaverTails dataset',loose=True)

def test_ambiguous_acronym_expansion_abstains():
    result=merge_acronyms({'ram':2,'random access memory':5,'recurrent attention model':3})
    assert 'ram' not in result
