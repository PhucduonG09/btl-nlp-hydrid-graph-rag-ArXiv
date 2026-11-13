"""Read-only assistant suggestions, separate from the blind human-rating form."""
import hashlib,json

def load_review(root):
    base=root/'data/eval/generation_calibration_v2';out=base/'assistant_review_v1'
    report=json.loads((out/'review.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((base/'blind_items.jsonl').read_bytes()).hexdigest()==report['input_pack_sha256'],'Source pack changed'
    assert hashlib.sha256((out/'decisions.json').read_bytes()).hexdigest()==report['decisions_sha256'],'Assistant decisions changed; compile again'
    assert report['n_reviewed']==50 and report['human_labels_written']==0
    items={r['sample_id']:r for r in map(json.loads,(base/'blind_items.jsonl').read_text(encoding='utf-8').splitlines())}
    assert len(report['items'])==len(items)==50
    return report,items

def render(root):
    import streamlit as st
    st.title('Nhận xét sau khi rà 50 mẫu')
    st.caption('Nhận xét của trợ lý theo nguồn · chỉ đọc · không phải nhãn người độc lập · không gọi API')
    try:report,items=load_review(root)
    except (OSError,ValueError,AssertionError) as error:st.error(f'Chưa có bản rà hợp lệ: {error}');return
    c=report['quality_counts']
    for column,(label,value) in zip(st.columns(4),[('Đáp ứng ý chính',c['correct']),('Một phần',c['partial']),('Lạc đích',c['incorrect']),('Từ chối',c['abstained'])]):column.metric(label,value)
    st.caption('Từ chối hợp lý khi thiếu nguồn vẫn không được tính là đáp án đúng. Các lỗi trong đầu ra gốc được giữ nguyên làm kết quả thí nghiệm.')
    with st.sidebar:
        group=st.selectbox('Phạm vi xem',['5 mẫu ưu tiên','5 góp ý nhỏ','Tất cả 50 mẫu'],key='assistant_review_group')
        selected=[r for r in report['items'] if group=='Tất cả 50 mẫu' or r['priority']==('high' if group=='5 mẫu ưu tiên' else 'low')]
        sample=st.selectbox('Mẫu',[r['sample_id'] for r in selected],key='assistant_review_sample')
        st.link_button('Mở biểu mẫu chấm người','http://127.0.0.1:8503/')
    r=next(r for r in selected if r['sample_id']==sample);item=items[sample]
    q={'correct':'Đáp ứng ý chính','partial':'Đúng một phần','incorrect':'Lạc đích câu hỏi','abstained':'Từ chối trả lời'}
    s={'all_supported':'Các khẳng định có nguồn','mixed':'Có ý vượt quá nguồn','unsupported':'Thiếu căn cứ','not_applicable':'Không chấm hỗ trợ dương cho đáp án từ chối'}
    st.subheader(f"{sample} · {q[r['quality']]}")
    st.write(s[r['support']]);st.markdown('**Nhận xét:** '+r['note'])
    if r.get('question_needs_review'):st.warning('Cần Hill chốt tính duy nhất của câu hỏi gốc. Bản frozen chưa bị sửa.')
    if r.get('proposed_revision'):
        with st.container(border=True):
            st.markdown('**Đề xuất, chưa ghi đè đầu ra thí nghiệm**');st.write(r['proposed_revision'])
    st.markdown('**Câu hỏi**');st.write(item['question'])
    st.markdown('**Đáp án gốc cần đánh giá**');st.write(item['candidate_answer'])
    with st.expander('Đáp án tham chiếu'):st.write(item['reference_answer'])
    with st.expander('Các đoạn nguồn đã đối chiếu'):
        for e in r['verified_source_spans']:st.write(f"[{e['source_number']}] arXiv:{e['paper_id']} — {e['text']}")
    for source in item['sources']:
        with st.expander(f"[{source['number']}] {source['title']}"):st.text(source['abstract'])
