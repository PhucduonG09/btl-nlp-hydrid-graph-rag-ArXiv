from src.kg.rule_extract import extract_one

def paper(abstract):return dict(paper_id='test',title='A method',abstract=abstract)
def pairs(p):return {(r['relation'],r['target']) for r in extract_one(p)['relations']}

def test_rule_distinguishes_own_method_component_and_baseline():
    result=pairs(paper('We introduce NewAgent using BaseModel. We compare against OldModel.'))
    assert ('PROPOSES','NewAgent') in result
    assert ('USES','BaseModel') in result
    assert not any(t=='OldModel' for _,t in result)

def test_rule_preserves_literal_evidence_and_does_not_use_negated_component():
    p=paper('We do not use OldModel. We evaluate on BenchSet dataset.')
    out=extract_one(p);text=p['title']+'. '+p['abstract']
    assert not any(r['target']=='OldModel' for r in out['relations'])
    assert ('EVALUATES_ON','BenchSet') in pairs(p)
    assert all(text[r['evidence_start']:r['evidence_end']]==r['evidence'] for r in out['relations'])
