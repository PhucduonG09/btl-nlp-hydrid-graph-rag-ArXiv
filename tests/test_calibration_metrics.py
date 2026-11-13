import pytest
from src.eval.calibration_metrics import agreement_metrics,judge_support_label

def test_empty_human_labels_do_not_produce_a_fake_perfect_agreement():
    assert agreement_metrics([],['yes','no'])['agreement'] is None
    assert agreement_metrics([('yes','yes')],['yes','no'])['cohen_kappa'] is None

def test_chance_agreement_and_macro_f1():
    result=agreement_metrics([('yes','yes'),('yes','no'),('no','yes'),('no','no')],['yes','no'])
    assert result['agreement']==.5 and result['cohen_kappa']==0 and result['macro_f1']==.5
    with pytest.raises(ValueError):agreement_metrics([('yes','unknown')],['yes','no'])

def test_abstention_is_not_all_supported_even_if_judge_emits_absence_claim():
    row=dict(quality='abstained',faithfulness_raw=1,claims=[dict(supported=True)])
    assert judge_support_label(row)=='not_applicable'
    row.update(quality='partial',claims=[dict(supported=True),dict(supported=False)])
    assert judge_support_label(row)=='mixed'
