"""Local benchmark review UI; saves only an independent SQLite decisions file."""
import json
from pathlib import Path
import streamlit as st
from src.eval.benchmark_review import decisions,effective,save,content_status

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'data/eval/benchmark_v2_draft'
DB=BASE/'review_decisions.sqlite3'
st.set_page_config(page_title='Duyệt benchmark NLP',layout='wide')
st.title('Duyệt benchmark cuối kỳ')
st.caption('150 ứng viên · 45 bridge giữ nguyên · lưu chỉnh sửa trực tiếp trên máy')
if notice:=st.session_state.pop('saved_notice',None):st.success(notice)

@st.cache_data(max_entries=2)
def read_jsonl(path,modified):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

candidate_path=BASE/'qa_candidates.jsonl'
if not candidate_path.exists():
    st.info('Chạy scripts/prepare_benchmark_v2.py rồi scripts/compile_benchmark_v2_draft.py trước.')
    st.stop()
raw=read_jsonl(str(candidate_path),candidate_path.stat().st_mtime_ns)
stored=decisions(DB);view={r['id']:effective(r,stored) for r in raw};original={r['id']:r for r in raw}
audit_path=BASE/'content_review_v2.json'
if not audit_path.exists():audit_path=BASE/'content_review_v1.json'
audit=json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}
audited={r['id']:r for r in audit.get('rows',[])}
statuses={qid:content_status(row,audited) for qid,row in view.items()}
case_path=BASE/'hill_decisions_needed.json'
cases={r['id']:r for r in json.loads(case_path.read_text(encoding='utf-8'))} if case_path.exists() else {}
approved=sum(r['review_status'] in {'approved','approved_existing'} for r in view.values())
with st.container(horizontal=True):
    st.metric('Ứng viên',len(raw))
    st.metric('Đã rà nội dung',sum(s in {'cleared','confirmed'} for s in statuses.values()))
    st.metric('Cần chốt / kiểm tra',sum(s not in {'cleared','confirmed'} for s in statuses.values()))
st.caption('Bộ này đang là nháp, chưa đóng băng làm test. Câu cũ được giữ provenance riêng vì đã xuất hiện trong thực nghiệm trước.')
st.caption('Dấu rà nội dung và nhật ký người duyệt được lưu riêng. Hill chỉ cần xử lý những câu còn vướng; không phải duyệt lại toàn bộ danh sách.')
filters={
    'Nháp mới':lambda r:r['origin']=='new_authored_draft',
    'Câu cũ có đề xuất sửa':lambda r:'revision_reason' in r,
    'Toàn bộ câu single cũ':lambda r:r['origin']=='legacy_single',
    'Bridge đã duyệt':lambda r:r['origin']=='bridge_v1_approved',
    'Tất cả':lambda r:True,
}
if any('replacement_approval' in r for r in raw):
    filters={'Các câu đã thay':lambda r:'replacement_approval' in r,**filters}
if audited and any(s not in {'cleared','confirmed'} for s in statuses.values()):
    filters={'Cần Hill chốt':lambda r:statuses[r['id']] not in {'cleared','confirmed'},**filters}
with st.sidebar:
    group=st.selectbox('Nhóm câu',list(filters),key='review_group')
    ids=[r['id'] for r in raw if filters[group](r)]
    if not ids:
        st.success('Nhóm này không còn câu cần xử lý.');st.stop()
    qid=st.selectbox('Mã câu',ids,key='review_id')
row=view[qid];source=original[qid]
if group=='Toàn bộ câu single cũ':
    unchanged=[r for r in raw if r['origin']=='legacy_single' and 'revision_reason' not in r
               and view[r['id']]['review_status'] not in {'approved','rejected'}
               and all(view[r['id']][k]==r[k] for k in ['question','reference_answer'])]
    if unchanged:
        with st.expander('Xác nhận các câu cũ đã được nhóm duyệt trước đây'):
            st.caption('Chỉ dùng nếu nhóm đã thực sự duyệt các câu cũ giữ nguyên. Sáu đề xuất sửa phải được xác nhận riêng. Nếu chưa duyệt, đọc từng câu ở dưới.')
            with st.form('legacy_confirmation'):
                legacy_reviewer=st.text_input('Người xác nhận',key='legacy_reviewer')
                attested=st.checkbox(f'Tôi xác nhận nhóm đã duyệt {len(unchanged)} câu cũ giữ nguyên này trước đây.',key='legacy_attested')
                batch=st.form_submit_button('Ghi nhận lần duyệt trước',key='legacy_confirm')
            if batch:
                if not attested or not legacy_reviewer.strip():st.error('Cần tên người xác nhận và xác nhận việc nhóm đã duyệt trước đây.')
                else:
                    for item in unchanged:
                        save(DB,item,question=item['question'],answer=item['reference_answer'],status='approved',reviewer=legacy_reviewer,
                             note='Người xác nhận khai báo nhóm đã duyệt trước đây; câu cũ giữ nguyên. Không phải một lượt gán nhãn độc lập mới.')
                    st.session_state['saved_notice']='Đã ghi nhận provenance duyệt trước cho '+str(len(unchanged))+' câu cũ giữ nguyên.'
                    st.rerun()
st.subheader(qid)
st.text('Trạng thái: '+row['review_status'])
if qid in audited:st.caption('Rà nguồn: '+audited[qid]['note'])
if 'replacement_approval' in row:st.success('Đã áp dụng câu thay thế theo lựa chọn của Hill, gồm paper và bằng chứng mới.')
if row.get('revision_reason'):
    st.info(row['revision_reason'])
    with st.expander('Nội dung cũ để đối chiếu'):
        prior=row.get('superseded',{})
        st.markdown('**Câu hỏi:** '+prior.get('question',row.get('previous_question',row['question'])))
        st.markdown('**Đáp án:** '+prior.get('reference_answer',row.get('previous_reference_answer',row['reference_answer'])))

corpus_path=ROOT/'data/processed/benchmark_v2_corpus.jsonl'
papers={p['paper_id']:p for p in read_jsonl(str(corpus_path),corpus_path.stat().st_mtime_ns)}
for pid in row['gold_ids']:
    paper=papers[pid]
    st.markdown('**'+paper['title']+'**')
    st.link_button('Mở paper '+pid,'https://arxiv.org/abs/'+pid)
    st.text(paper['abstract'])
if row.get('evidence_spans'):
    with st.expander('Đoạn bằng chứng cho bản nháp'):
        for span in row['evidence_spans']:st.text(span['text'])

if row['origin']=='bridge_v1_approved':
    st.markdown('**Câu hỏi:** '+row['question'])
    st.markdown('**Đáp án:** '+row['reference_answer'])
    st.caption('Bridge đã duyệt được khóa ở giao diện này. File Excel và bridge v1 không bị sửa.')
elif qid in cases and statuses[qid]=='needs_hill_decision':
    case=cases[qid];proposal=case['replacement']
    st.warning(case['issue'])
    st.markdown('**Câu hiện tại:** '+row['question'])
    st.markdown('**Đáp án hiện tại:** '+row['reference_answer'])
    st.markdown('**Đề xuất:** '+case['recommendation'])
    with st.expander('Xem câu thay thế đã chuẩn bị',expanded=True):
        st.markdown('**'+proposal['title']+'**')
        st.link_button('Mở paper thay thế', 'https://arxiv.org/abs/'+proposal['paper_id'])
        st.markdown('**Câu mới:** '+proposal['question'])
        st.markdown('**Đáp án mới:** '+proposal['reference_answer'])
        st.text(proposal['abstract'])
        for span in proposal['evidence_spans']:st.text(span['text'])
    st.info('Chưa áp dụng câu thay thế. Hill có thể chốt ngay trong chat: “thay '+qid+'”, “giữ '+qid+'” hoặc “thay cả 3”. Mình sẽ cập nhật đúng paper và bằng chứng đi kèm.')
else:
    with st.form('decision_'+qid):
        question=st.text_area('Câu hỏi',value=row['question'],height=120,key='q_'+qid)
        answer=st.text_area('Đáp án',value=row['reference_answer'],height=170,key='a_'+qid)
        reviewer=st.text_input('Tên người duyệt',value=row.get('reviewer',''),key='reviewer_'+qid)
        choice=st.radio('Quyết định',['Chưa chốt','Đạt','Loại'],index={'pending':0,'approved':1,'rejected':2}.get(row['review_status'],0),key='choice_'+qid,horizontal=True)
        note=st.text_area('Ghi chú',value=row.get('review_note',''),key='note_'+qid)
        submitted=st.form_submit_button('Lưu trên máy',key='save_'+qid)
    if submitted:
        try:
            save(DB,source,question=question,answer=answer,status={'Chưa chốt':'pending','Đạt':'approved','Loại':'rejected'}[choice],reviewer=reviewer,note=note)
        except ValueError as error:st.error(str(error))
        else:
            st.session_state['saved_notice']='Đã lưu quyết định cho '+qid+' trên máy.'
            st.rerun()
st.caption('Nút lưu chỉ ghi review_decisions.sqlite3 trong thư mục benchmark_v2_draft. Không ghi vào workbook, QA cũ hay tập test v1.')
