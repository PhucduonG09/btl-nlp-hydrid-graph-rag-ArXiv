from copy import deepcopy
import pytest
from src.eval.benchmark_replacements import apply_replacements
from src.eval.benchmark_review import fingerprint

def fixture():
    row=dict(id='q',type='single',question='Old?',reference_answer='Old answer',gold_ids=['old'],origin='legacy_single')
    paper=dict(title='New method',abstract='We use token identifiers.')
    change=dict(id='q',superseded_fingerprint=fingerprint(row),approval=dict(reviewer='unit-test'),
                replacement=dict(paper_id='new',**paper,question='New?',reference_answer='Token identifiers.',evidence=['We use token identifiers.']))
    return row,paper,change

def test_replacement_updates_gold_and_source_evidence_together():
    row,paper,change=fixture();before=deepcopy(row)
    result=apply_replacements([row],[change],{'new':paper})[0]
    assert row==before and result['gold_ids']==['new']
    assert result['superseded']['gold_ids']==['old']
    assert result['origin']=='new_authored_draft'
    span=result['evidence_spans'][0];source=paper['title']+'. '+paper['abstract']
    assert source[span['start']:span['end']]=='We use token identifiers.'

@pytest.mark.parametrize('bad',['bridge','stale','duplicate','evidence'])
def test_replacement_rejects_invalid_source_or_evidence(bad):
    row,paper,change=fixture();rows=[row]
    if bad=='bridge':row['origin']='bridge_v1_approved';change['superseded_fingerprint']=fingerprint(row)
    if bad=='stale':row['question']='Changed draft?'
    if bad=='duplicate':rows.append(dict(id='another',type='single',gold_ids=['new']))
    if bad=='evidence':change['replacement']['evidence']=['An invented result']
    with pytest.raises(AssertionError):apply_replacements(rows,[change],{'new':paper})
