from src.eval.run_status import summarize_records,read_records

def test_usage_counts_paid_attempts_but_not_cache_hits():
    records=[dict(id='a',system='x',error='parse',usage={'in':10,'out':2,'cached':False}),
             dict(id='a',system='x',usage={'in':15,'out':3,'cached':False}),
             dict(id='b',system='x',usage={'in':20,'out':4,'cached':True})]
    summary=summarize_records(records)
    assert summary['completed']==2 and summary['errors']==0
    assert summary['provider_input_tokens']==25 and summary['provider_output_tokens']==5

def test_progress_reader_tolerates_a_partially_written_tail(tmp_path):
    path=tmp_path/'progress.jsonl';path.write_text('{"id":"a"}\n{"id":',encoding='utf-8')
    records,bad=read_records(path)
    assert len(records)==1 and bad==1
