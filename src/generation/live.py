"""One explicit user query, same generator and prompt contract as DEV."""
import json,os,threading
from datetime import datetime,timedelta,timezone
from pathlib import Path
from src.generation.controlled import prepare,citation_diagnostics
ROOT=Path(__file__).resolve().parents[2]
LOCK=threading.Lock()

def ask(engine,question,method):
    if len(question.strip())<10:raise ValueError('Nhập câu hỏi dài ít nhất 10 ký tự.')
    os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT/'data/cache/tiktoken')
    import tiktoken
    from src.common import llm
    retrieval=engine.retrieve(question,method)
    payload=prepare(question,retrieval['ranked'],engine.papers,tiktoken.get_encoding('cl100k_base'),budget=6000,top_k=10)
    # This Streamlit process has its own cache path; benchmark workers use other files.
    with LOCK:
        cache=ROOT/'results/demo_v2/llm_cache.jsonl'
        if llm.CACHE_PATH!=cache:llm.CACHE_PATH=cache;llm._cache=None
        text,usage=llm.chat(payload['messages'],model='claude-sonnet-5',temperature=0,max_tokens=768,run_id='live-demo-v2')
    result=dict(question=question,method=method,answer=text,usage=usage,sources=payload['sources'],retrieval=retrieval,
                **citation_diagnostics(text,payload['sources']))
    log=ROOT/'results/demo_v2/queries.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
    with LOCK:
        with log.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(recorded_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),**result),ensure_ascii=False)+'\n')
    return result
