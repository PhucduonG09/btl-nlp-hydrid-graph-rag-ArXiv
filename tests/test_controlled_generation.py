from src.generation.controlled import prepare,citation_diagnostics

class Chars:
    def encode(self,text,**kwargs):return list(text)
    def decode(self,tokens):return ''.join(tokens)

def test_context_cap_deduplication_and_reference_not_in_messages():
    papers={'a':dict(title='Title',abstract='x'*400),'b':dict(title='Other',abstract='y'*400)}
    out=prepare('Question?',['a','a','b'],papers,Chars(),budget=180)
    assert out['context_proxy_tokens']<=180
    assert len({s['paper_id'] for s in out['sources']})==len(out['sources'])
    assert out['sources'][0]['truncated']
    assert 'reference_answer' not in str(out['messages'])

def test_valid_citations_are_not_scored_as_faithful():
    out=citation_diagnostics('A claim [1] and another [9].',[dict(number=1,paper_id='a')])
    assert out['invalid_citations']==[9] and out['cited_papers']==['a']
    assert out['faithfulness'] is None
