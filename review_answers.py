"""Blind human calibration UI. No automated scores, system identities or API calls."""
import hashlib,json,os
from pathlib import Path
import streamlit as st
from src.eval.human_calibration import ratings,save_rating,resolved_ratings
ROOT=Path(__file__).resolve().parent;BASE=ROOT/'data/eval/generation_calibration_v2';DB=BASE/'ratings.sqlite3'
st.set_page_config(page_title='Chấm đáp án NLP',layout='wide')
SHARED=os.environ.get('REVIEW_SHARED_MODE')=='1'
if not SHARED and st.query_params.get('view')=='assistant':
    from src.eval.assistant_answer_review_ui import render
    render(ROOT)
    st.stop()
st.title('Chấm đáp án theo nguồn')
st.caption('50 mẫu · ẩn tên hệ thống và điểm máy chấm · lưu trực tiếp trên máy · không gọi API')
if not SHARED and (BASE/'assistant_review_v1/review.json').exists():
    st.link_button('Xem 50 nhận xét đã rà bởi trợ lý','http://127.0.0.1:8503/?view=assistant')
    st.caption('Bản rà của trợ lý nằm ở trang riêng. Tiến độ dưới đây chỉ đếm các nhãn do người chấm nhập.')
path=BASE/'blind_items.jsonl'
if not path.exists():st.info('Gói chấm chưa được chuẩn bị.');st.stop()
manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['items_sha256']:st.error('Nội dung gói chấm đã thay đổi.');st.stop()
items=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
stored=ratings(DB);completed,pending=resolved_ratings(stored,{i['sample_id'] for i in items})
st.progress(len(completed)/len(items),text=f'{len(completed)}/{len(items)} mẫu có nhãn người đã chốt')
with st.expander('Cách chấm',expanded=False):
    st.markdown('Đọc câu hỏi, đáp án và các abstract được cung cấp. **Đúng** khi trả lời đủ các phần với thông tin chính xác; **Đúng một phần** khi thiếu ý quan trọng; **Sai** khi sai nội dung chính; **Từ chối** khi đáp án không trả lời vì thiếu thông tin. Đáp án tham chiếu cũng có thể chưa hoàn hảo: đánh dấu nếu thấy vấn đề.')
    st.markdown('Chấm **hỗ trợ nguồn** riêng: mọi khẳng định có căn cứ / có ý không được nguồn hỗ trợ / không có căn cứ. Một câu trả lời có thể đúng theo kiến thức bên ngoài nhưng vẫn thiếu căn cứ trong nguồn hiển thị. Không áp dụng hỗ trợ nguồn cho đáp án chỉ từ chối.')
with st.sidebar:
    reviewer=st.text_input('Tên người chấm',key='human_reviewer')
    if SHARED:st.caption('Mỗi bạn dùng một tên riêng và giữ nguyên tên đó trong các lần chấm sau.')
    sample_id=st.selectbox('Mẫu',[i['sample_id'] for i in items],key='human_sample')
item=next(i for i in items if i['sample_id']==sample_id)
st.subheader(item['question']);st.markdown('**Đáp án cần chấm**');st.markdown(item['candidate_answer'])
with st.expander('Đáp án tham chiếu'):st.markdown(item['reference_answer'])
for source in item['sources']:
    with st.expander(f"[{source['number']}] {source['title']}"):
        st.text(source['abstract'])
q_labels={'Chưa chọn':'','Đúng':'correct','Đúng một phần':'partial','Sai':'incorrect','Từ chối':'abstained','Chưa đủ căn cứ để chốt':'uncertain'}
s_labels={'Chưa chọn':'','Mọi ý đều được nguồn hỗ trợ':'all_supported','Có ý chưa được nguồn hỗ trợ':'mixed','Không được nguồn hỗ trợ':'unsupported','Không áp dụng — đáp án từ chối':'not_applicable','Chưa đủ căn cứ để chốt':'uncertain'}
prior=next((r for r in stored if r['sample_id']==sample_id and r['reviewer']==reviewer.strip()),{})
with st.form('human_rating_'+sample_id+'_'+reviewer):
    quality=st.selectbox('Mức trả lời câu hỏi',list(q_labels),index=next((i for i,k in enumerate(q_labels) if q_labels[k]==prior.get('quality')),0))
    support=st.selectbox('Mức hỗ trợ từ nguồn',list(s_labels),index=next((i for i,k in enumerate(s_labels) if s_labels[k]==prior.get('support')),0))
    reference_issue=st.checkbox('Câu hỏi hoặc đáp án tham chiếu cần xem lại',value=bool(prior.get('reference_issue')))
    note=st.text_area('Ghi chú ý sai/thiếu và nguồn liên quan',value=prior.get('note',''))
    submit=st.form_submit_button('Lưu nhãn người',key='save_human_rating')
if submit:
    try:save_rating(DB,sample_id=sample_id,reviewer=reviewer,quality=q_labels[quality],support=s_labels[support],note=note,reference_issue=reference_issue)
    except ValueError as error:st.error(str(error))
    else:st.success('Đã lưu trên máy. Bảng tiến độ cập nhật khi chuyển mẫu hoặc tải lại.')
