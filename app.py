"""Read-only DEV evidence explorer. Never reads test QA or calls a paid API."""
import json
from pathlib import Path
import streamlit as st

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Hybrid Graph-RAG ArXiv',layout='wide')
st.title('Hybrid Graph-RAG trên ArXiv')
st.caption('Khám phá bằng chứng và hỏi đáp trên kho paper ArXiv')

with st.sidebar:
    mode=st.radio('Chế độ',['Xem thực nghiệm','Hỏi đáp','Tiến độ chạy'],key='mode')
if mode=='Tiến độ chạy':
    from src.eval.run_status import read_records,summarize_records
    st.subheader('Tiến độ lượt gọi LLM trên DEV')
    @st.fragment(run_every='20s')
    def execution_progress():
        folder=ROOT/'results/benchmark_v2_assisted/dev_01/generation_prepared/claude5_controlled'
        rows=[]
        for name,file in [('Sinh câu trả lời','generation.jsonl'),('LLM-judge','judge.jsonl')]:
            records,bad=read_records(folder/file);summary=summarize_records(records)
            st.markdown('**'+name+'**')
            st.progress(min(summary['completed']/244,1.),text=f"{summary['completed']}/244 · lỗi: {summary['errors']}")
            rows.append(dict(Giai_đoạn=name,Token_vào=summary['provider_input_tokens'],Token_ra=summary['provider_output_tokens'],Lệch_nhãn_model=summary['model_label_mismatches']))
            if bad:st.caption(f'{bad} dòng log chưa đọc được; có thể có dòng đang ghi.')
        st.table(rows,alt='Token do API báo sử dụng ở hai giai đoạn của lượt DEV chính')
        st.caption('Tự cập nhật mỗi 20 giây. Token là số API báo trong các phản hồi đã ghi, gồm các lượt thử lại có usage và loại lượt trúng cache; chưa tính tiền vì chưa có đơn giá. Pilot ban đầu có nhãn model lẫn và các probe được lưu riêng.')
        st.caption('Bảng tiến độ chỉ đọc log, không gọi thêm LLM. Test chưa được chạy.')
    execution_progress();st.stop()
if mode=='Hỏi đáp':
    st.subheader('Hỏi đáp trên kho 20.500 paper')
    st.caption('Khi bấm Trả lời, câu hỏi và tối đa 10 abstract được gửi tới LLM. Context và model dùng cùng cấu hình với thực nghiệm DEV.')
    with st.form('live_question'):
        question=st.text_area('Câu hỏi',key='live_question_text')
        live_method=st.selectbox('Truy xuất',['graph','rrf'],key='live_method')
        submitted=st.form_submit_button('Trả lời',key='live_submit')
    @st.cache_resource(max_entries=1)
    def live_engine():
        from src.retrieval.engine import load_engine
        return load_engine()
    if submitted:
        if len(question.strip())<10:st.error('Nhập câu hỏi dài ít nhất 10 ký tự.')
        else:
            try:
                with st.spinner('Đang truy xuất và tạo câu trả lời…'):
                    from src.generation.live import ask
                    st.session_state['live_result']=ask(live_engine(),question.strip(),live_method)
            except Exception as error:st.error('Chưa tạo được câu trả lời: '+type(error).__name__)
    result=st.session_state.get('live_result')
    if result:
        st.markdown(result['answer'])
        usage=result['usage']
        st.caption(f"Model trả về: {usage.get('served_model')} · token vào/ra: {usage.get('in')}/{usage.get('out')} · cache: {usage.get('cached')}")
        if result['invalid_citations']:st.warning('Có chỉ số trích dẫn không hợp lệ: '+str(result['invalid_citations']))
        for source in result['sources']:
            with st.expander(f"[{source['number']}] {source['title']}"):
                st.link_button('Mở paper '+source['paper_id'],'https://arxiv.org/abs/'+source['paper_id'])
                st.text(source['abstract'])
                for edge in result['retrieval']['paths'].get(source['paper_id'],[]):
                    st.caption(f"Đường nhắc tên: {edge['A']} → {edge['entity']} ← {edge['B']} (MENTIONS)")
    st.stop()
with st.sidebar:
    version=st.selectbox('Phiên bản',['Benchmark v2 · DEV','Bridge v1 · DEV'],key='dataset')
st.caption('Chế độ xem thực nghiệm chỉ đọc kết quả đã lưu, không gọi API.')

@st.cache_data(max_entries=2)
def load_data(version):
    def rows(path):return [json.loads(l) for l in (ROOT/path).read_text(encoding='utf-8').splitlines() if l.strip()]
    modern=version.startswith('Benchmark v2')
    base='benchmark_v2_assisted' if modern else 'bridge_v1'
    run_dir='results/benchmark_v2_assisted/dev_01' if modern else 'results/bridge_v1/graph_dev'
    qa=rows(f'data/eval/{base}/qa_dev.jsonl')
    runs={r['id']:r for r in rows(run_dir+'/retrieval.jsonl')}
    corpus='benchmark_v2_corpus' if modern else 'bridge_corpus_v1'
    papers={p['paper_id']:p for p in rows(f'data/processed/{corpus}.jsonl')}
    graph=json.loads((ROOT/('data/kg/reference_mentions_v2_assisted.json' if modern else 'data/kg/reference_mentions_v1.json')).read_text(encoding='utf-8'))
    metrics=json.loads((ROOT/run_dir/'metrics.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/run_dir/'selected_config.json').read_text(encoding='utf-8'))
    return qa,runs,papers,graph,metrics,config

modern=version.startswith('Benchmark v2')
required=ROOT/('results/benchmark_v2_assisted/dev_01/retrieval.jsonl' if modern else 'results/bridge_v1/graph_dev/retrieval.jsonl')
if not required.exists():
    st.info('Chạy scripts/build_reference_graph.py và scripts/run_graph_dev.py trước khi mở bản demo.')
    st.stop()
qa,runs,papers,graph,metrics,config=load_data(version)
with st.sidebar:
    qid=st.selectbox('Câu hỏi DEV',[q['id'] for q in qa],key='question_id')
    methods=list(runs[qid]['rankings'])
    if modern:
        advanced=st.toggle('Xem toàn bộ cấu hình',key='advanced')
        if not advanced:methods=[config['bm25'],'dense',config['rrf'],'linked_text_expansion',config['graph']]
    method=st.selectbox('Phương pháp',methods,index=methods.index(config['graph']) if modern else 0,key='method')
    show_gold=st.toggle('Đối chiếu đáp án tham chiếu',key='show_gold')
q=next(q for q in qa if q['id']==qid);run=runs[qid]
st.subheader(q['question'])
st.caption('Đồ thị MENTIONS ghi nhận tên thực thể trong tiêu đề/tóm tắt. Quan hệ này không khẳng định paper sử dụng hay đề xuất thực thể.')
with st.container(horizontal=True):
    st.metric('Kho bài báo',f"{graph['stats']['papers']:,}")
    st.metric('All@10 trên DEV',f"{(metrics[method]['all'] if modern else metrics[method])['All@10']:.1%}")
    st.metric('Bridge: đủ hai paper',f"{(metrics[method]['bridge'] if modern else metrics[method])['All@10']:.1%}")
st.caption('Chỉ số trên DEV đã dùng chọn cấu hình. Các câu bridge được sàng lọc theo độ khó BM25; chưa phải kết quả test cuối kỳ.')
if modern:
    compare=[dict(Phương_pháp=m,Single_R10=f"{metrics[m]['single']['R@10']:.1%}",Bridge_All10=f"{metrics[m]['bridge']['All@10']:.1%}")
             for m in [config['bm25'],'dense',config['rrf'],'linked_text_expansion',config['graph']]]
    st.table(compare,alt='So sánh riêng kết quả truy xuất single và bridge trên DEV')
    st.caption('122 câu DEV của bản thực nghiệm có nguồn gán nhãn người/trợ lý được ghi riêng; đây chưa phải bộ 150 nhãn người độc lập.')
ranking=run['rankings'][method]
path_key=next((k for k in run.get('paths',{}) if k==method or method.startswith(k+'_rrf_') or method==k+'_weighted'),method)
paths=run.get('paths',{}).get(path_key,{})
for rank,pid in enumerate(ranking[:10],1):
    p=papers[pid]
    with st.expander(f"{rank}. {p['title']} ({pid})"):
        st.link_button('Mở bài báo',f'https://arxiv.org/abs/{pid}')
        st.text(p['abstract'])
        for edge in paths.get(pid,[]):
            entity=edge['entity']
            st.markdown(f"**Đường nối:** {edge['A']} → {graph['labels'][entity]} ← {edge['B']} (MENTIONS)")
            for endpoint in [edge['A'],edge['B']]:
                source=papers[endpoint]['title']+'. '+papers[endpoint]['abstract']
                span=graph['evidence'][endpoint][entity]
                st.caption(f"{endpoint}: …{source[max(0,span['start']-100):span['end']+160]}…")
if show_gold:
    st.subheader('Đối chiếu dữ liệu đã duyệt')
    st.markdown(q['reference_answer'])
    targets=[(label,q[label]) for label in ['hop1_id','hop2_id']] if q['type']=='bridge' else [('gold_id',pid) for pid in q['gold_ids']]
    for label,pid in targets:
        st.text(f"{label}: {pid} · hạng {ranking.index(pid)+1 if pid in ranking else '>100'}")
st.info('Chọn Hỏi đáp để đặt câu mới, hoặc Tiến độ chạy để xem lượt đánh giá LLM. Các kết quả trên trang này là DEV; đánh giá độc lập bởi người vẫn được theo dõi riêng.')
