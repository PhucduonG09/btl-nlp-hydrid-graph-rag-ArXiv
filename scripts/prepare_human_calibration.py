"""Freeze 50 blind answer-review items, sampled without looking at judge labels."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    base=ROOT/'results/benchmark_v2_assisted/dev_01/generation_prepared';gen_path=base/'claude5_controlled/generation.jsonl'
    generated={(r['id'],r['system']):r for r in load(gen_path)}
    qa_path=ROOT/'data/eval/benchmark_v2_assisted/qa_dev.jsonl';qa=load(qa_path)
    assert len(generated)==244 and all(r.get('answer') and not r.get('error') for r in generated.values())
    jobs={(r['id'],r['system']):r for r in load(base/'jobs_dev.jsonl')}
    rng=random.Random(20261109);selected=[]
    for kind in ['single','bridge']:selected+=rng.sample([q for q in qa if q['type']==kind],25)
    rng.shuffle(selected);systems=['rrf','graph']*25;rng.shuffle(systems)
    items=[];key=[]
    for i,(q,system) in enumerate(zip(selected,systems),1):
        sample=f'H{i:03d}';record=generated[(q['id'],system)];job=jobs[(q['id'],system)]
        items.append(dict(sample_id=sample,question=q['question'],candidate_answer=record['answer'],reference_answer=q['reference_answer'],sources=job['sources']))
        key.append(dict(sample_id=sample,question_id=q['id'],system=system,type=q['type']))
    out=ROOT/'data/eval/generation_calibration_v2';out.mkdir(exist_ok=True)
    item_path=out/'blind_items.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items)
    if item_path.exists():assert item_path.read_text(encoding='utf-8')==text,'Blind pack changed: create a new version'
    else:item_path.write_text(text,encoding='utf-8')
    key_path=base/'claude5_controlled/human_calibration_key.json'
    if key_path.exists():assert json.loads(key_path.read_text(encoding='utf-8'))==key
    else:key_path.write_text(json.dumps(key,indent=2),encoding='utf-8')
    manifest=dict(n_items=50,n_distinct_questions=len({r['question_id'] for r in key}),single=25,bridge=25,rrf=25,graph=25,seed=20261109,
                  items_sha256=sha(item_path),generation_file_sha256=sha(gen_path),qa_dev_sha256=sha(qa_path),
                  selection='Random stratified sample, no judge labels or scores used. One response per distinct question; systems hidden in review UI.',
                  ratings_written=0,independent_human_calibration_complete=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
