"""Produce a DEV gap/model/evidence audit without changing predictions or graph."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.kg.prompts import PROMPTS
from src.kg.llm_extract import parse_json,_ws,VALID_REL

def load(path):
    with path.open(encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]

def main():
    out=ROOT/'results/bridge_v1'
    records={r['paper_id']:r for r in load(ROOT/'data/kg/bridge_dev_v1_extract_v3.jsonl')}
    papers={p['paper_id']:p for p in load(ROOT/'data/processed/bridge_v1_dev_missing_kg_papers.jsonl')}
    qa={q['id']:q for q in load(ROOT/'data/eval/bridge_v1/qa_dev.jsonl')}
    runtime=json.loads((out/'extraction_full_runtime.json').read_text(encoding='utf-8'))
    requested=runtime['requested_model'];keys={}
    for pid,p in papers.items():
        messages=[{'role':'system','content':PROMPTS['v3']},{'role':'user','content':f"Title: {p['title']}\nAbstract: {p['abstract']}"}]
        key=hashlib.sha256(json.dumps([requested,messages,{'temperature':0.0,'max_tokens':1800},0],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        keys[key]=pid
    cached={}
    with (ROOT/'data/cache/llm.jsonl').open(encoding='utf-8') as f:
        for line in f:
            try:r=json.loads(line)
            except ValueError:continue
            if r.get('key') in keys:cached[keys[r['key']]]=r
    rejects=[]
    for pid,r in cached.items():
        p=papers[pid];text=f"Title: {p['title']}\nAbstract: {p['abstract']}"
        raw=parse_json(r['text'])
        if not isinstance(raw,dict):continue
        for edge in raw.get('relations',[]):
            if not isinstance(edge,dict):continue
            rel=str(edge.get('relation','')).strip().upper();target=str(edge.get('target','')).strip();ev=str(edge.get('evidence','')).strip()
            if rel in VALID_REL and target and (not ev or _ws(ev) not in _ws(text)):
                rejects.append(dict(paper_id=pid,relation=rel,target=target,rejected_evidence=ev,
                                    action='Review verbatim source span and relation semantics; not automatically restored.'))
    expected_rejects=sum(r.get('dropped_no_evidence',0) for r in records.values())
    assert len(cached)==len(papers) and len(rejects)==expected_rejects
    with (out/'extraction_rejected_evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rejects[0]));w.writeheader();w.writerows(rejects)
    reasons={
      'bd001':('Tên/task khác cấp độ','A trích Neural Theory-of-Mind assessment và 6 N-ToM tasks; B trích Theory of Mind reasoning/higher-order tasks. Còn một contribution bị lọc evidence.', 'Rà khái niệm/task hierarchy theo ngữ cảnh; không gộp ToM với mọi chuỗi chứa tom.'),
      'bd002':('Thiếu loại quan hệ đánh giá model','A và B đang được trích theo nhiệm vụ/benchmark, không có cạnh tới model GPT-4V.', 'Cần quan hệ đánh giá model riêng hoặc quy tắc USES được gán nhãn rõ; không gộp GPT-4V thành GPT-4.'),
      'bn002':('Cạnh bị lọc do evidence','A Memento có 3 dự đoán bị lọc evidence, gồm cạnh benchmark; B đã PROPOSES GAIA.', 'Đối chiếu lại quote nguồn, đặc biệt markup LaTeX; chỉ khôi phục nếu quan hệ và span thật sự đúng.'),
      'bn003':('Sai vai trò và lệch tên','A gán PROPOSES DINOISER loss function; B PROPOSES DINOISER.', 'Rà novelty của A và quan hệ loss/phương pháp gốc. Không sửa bằng cách gộp mọi hậu tố tùy ý.'),
      'bn014':('Bỏ sót quan hệ tới dữ liệu gốc','A PROPOSES HelpSteer2-Preference nhưng không dự đoán cạnh tới HelpSteer2. Quote bị loại thuộc model Instruct, không phải cạnh dataset gốc.', 'Phân biệt dataset gốc với bản preference mở rộng; rà quy tắc USES khi bổ sung/biến đổi dataset.'),
      'bn020':('Sai vai trò và quote nguồn B','A gán PROPOSES CICLe dù là bài đánh giá framework cũ; contribution của B có một quote bị loại.', 'Rà lại USES/PROPOSES theo novelty và evidence nguyên văn. Đường qua Conformal Prediction chưa phải đường gold CICLe.'),
      'bn025':('Baseline không được biểu diễn','A cố ý không trích HIVE thành USES theo prompt v3; B PROPOSES HIVE.', 'Nếu giữ câu, cần COMPARES_TO có evidence và gold tương ứng.'),
      'bn034':('Quan hệ phòng vệ không được biểu diễn','A trích FilterRAG/ML-FilterRAG và nhiệm vụ knowledge poisoning, chưa có cạnh tới PoisonedRAG.', 'Nếu giữ câu, cần DEFENDS_AGAINST hoặc loại reference phù hợp, có evidence.'),
      'bn049':('Sự khác biệt tên và quan hệ sibling','B PROPOSES BeaverTails dataset; A chỉ có shared task safety alignment, không có cạnh sibling tới BeaverTails.', 'Chuẩn hóa tên chưa đủ; cần xác định quan hệ SIBLING_PROJECT/RELATED_DATASET theo source.'),
      'bn050':('Baseline không được biểu diễn','A không trích QLLM thành USES theo prompt v3; B PROPOSES QLLM.', 'Nếu giữ câu, cần COMPARES_TO có evidence và gold tương ứng.'),
      'bn011':('Sai novelty dù có đường đi','A gán PROPOSES KHRONOS trong khi mô tả sử dụng surrogate đã có; B cũng PROPOSES KHRONOS.', 'Rà quy tắc novelty và gold: có đường đi không chứng minh nhãn quan hệ đúng.'),
    }
    with (out/'kg_audit_dev_bridge_dev_v1_v3.csv').open(encoding='utf-8-sig') as f:audits={r['id']:r for r in csv.DictReader(f)}
    gaps=[]
    for qid,(reason,detail,action) in reasons.items():
        q=qa[qid]
        gaps.append(dict(id=qid,source=q['source'],source_row=q['source_row'],bridge_entity=q['bridge_entity'],
                         has_expected_path=audits[qid]['expected_bridge_two_edge_path'],reason=reason,detail=detail,next_action=action))
    with (out/'kg_gap_analysis_dev.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(gaps[0]));w.writeheader();w.writerows(gaps)
    drift=[pid for pid,r in records.items() if r['usage'].get('served_model') not in {'claude-sonnet-5-5','claude-sonnet-5.5'}]
    model_audit=dict(n_papers=len(records),requested_model=requested,
                     reported_models=dict(Counter(r['usage'].get('served_model','unknown') for r in records.values())),
                     reported_model_mismatch_papers=drift,
                     affected_dev_questions=[q['id'] for q in qa.values() if set(q['gold_ids'])&set(drift)],
                     interpretation='Served model is a provider-reported label, not independent verification of the actual model. Resolve version inconsistency before a controlled final comparison.',
                     successful_record_tokens_in=sum(r['usage']['in'] for r in records.values()),
                     successful_record_tokens_out=sum(r['usage']['out'] for r in records.values()),
                     billing_limit='Token totals cover successfully parsed records; responses lost to earlier transport decoding errors have no recorded usage. Provider billing is authoritative.',
                     rejected_evidence=len(rejects),raw_cache_records_found=len(cached),predictions_modified=False,KG_modified=False)
    (out/'extraction_model_audit.json').write_text(json.dumps(model_audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(model_audit,ensure_ascii=False,indent=2))
    print('Rejected evidence records:')
    for r in rejects:print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()
