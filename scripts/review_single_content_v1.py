"""Record this source-review pass. Does not write human decisions or workbooks.

The judgments below were authored during source reading, not inferred from rank.
Existing review fingerprints must still match before this snapshot can be reused.
"""
import hashlib,json,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.benchmark_review import fingerprint
from src.kg.evidence import locate_evidence
from scripts.compile_benchmark_v2_draft import main as compile_draft
BASE=ROOT/'data/eval/benchmark_v2_draft'
def load(path):return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

NEW_EDITS={
 's2007':dict(question='What operations does AMDM combine, and what latency and false-positive results are reported for its monitoring experiments?',reference_answer='AMDM normalizes heterogeneous metrics, applies per-axis exponentially weighted moving-average thresholds, and uses Mahalanobis distance for joint anomaly detection. On simulated goal drift, latency falls from 12.3 to 5.6 seconds. The reported false-positive rate falls from 4.5% to 0.9% compared with static thresholds.'),
 's2011':dict(question='Across the 24-model altruism study, how do self-reported and behavioral altruism differ, and what correlation between implicit associations and behavior is reported?',reference_answer='Models report 77.5% altruism but behave altruistically at 65.6%. The reported association between implicit measures and behavior is r = .22 with p = .29; the study concludes that implicit associations do not predict behavior.'),
 's2012':dict(reference_answer='The proposed WE-Clustering_K++ variant based on K-means reports a 45% increase, while the WEClustering_A++ agglomerative variant reports a 67% increase in median silhouette score.'),
 's2015':dict(question='In the phrase-mining method evaluated on an augmented STS-B dataset, why is a whole-sentence vector inadequate in noisy context, and what property should the trained token embeddings support?',reference_answer='A whole-sentence vector can fail to retrieve a target phrase embedded in noisy surrounding text. The contrastive-loss modification encourages contextualized word/token embeddings that can be aggregated into arbitrary consecutive spans while preserving the meaning of each span.'),
 's2016':dict(question='In the online-safety study benchmarking both open- and closed-source LLMs, when are risks assessed, and what hybrid analysis strategy is explored?'),
 's2029':dict(question='In the SWIM study of self-adaptation rules, what role do LLMs play, and how is the experimental evidence characterized?',reference_answer='LLMs act as optimizers that construct and improve adaptation rules using commonsense and reasoning. The abstract describes preliminary SWIM experiments demonstrating both effectiveness and limitations; it does not establish a universal improvement guarantee.'),
 's2036':dict(question='In the text-classification ensemble study using predictions from five fine-tuned models, what resource is assembled and how many datasets does it cover?'),
 's2039':dict(question='Does ExtAgents require longer-context training, and are its reported knowledge-integration improvements limited to inputs exceeding the context window?',reference_answer='ExtAgents scales inference-time knowledge integration without longer-context training. The reported gains over non-training alternatives use the same amount of external knowledge and apply whether it fits within or exceeds the context window; high parallelism helps maintain efficiency.'),
 's2041':dict(question='In the historical-semantics comparison of ChatGPT, GPT-4, Claude, Bard, Gemini and Llama, which automatic measures and human assessments are used?'),
 's2043':dict(question='In dual-perspective instruction learning with white-box and black-box LLMs, which model supplies initial instructions, which white-box features are used, and what constraint combines the components?'),
}
OLD_EDITS={
 'q018':dict(question='In the docking model built around TAMformer, how are whole-ligand graph features and three-dimensional geometry used to refine predicted binding poses?',reason='Làm rõ mô hình được hỏi bằng thành phần TAMformer có trong nguồn; không coi các cách docking bất kỳ là đáp án thay thế tương đương.'),
 'q020':dict(question='For an online predictor with epsilon distance to calibration, what ECE and Calibration Decision Loss can post-processing achieve, and how does that rate compare with direct optimization?',reason='Đặt rõ giả thiết và hai đại lượng đầu ra, tránh lẫn với bài toán hiệu chuẩn đa lớp hoặc các khái niệm smooth calibration khác.'),
 'q025':dict(question='How does STEER reshape logits at inference time to balance target-domain fidelity against repetition of real and previously generated examples?',reason='Giới hạn về STEER và hai nhánh logit reshaping có nguồn; câu cũ cho phép nhiều giải pháp sinh dữ liệu khác nhau.'),
 'q027':dict(question='How does contextualizing HTML elements with screenshot counterparts and neighboring elements improve action selection in the Mind2Web evaluation?',reason='Đặt rõ cách biểu diễn dual-view và benchmark có trong nguồn, thay cho câu hỏi chung về mọi web agent.'),
 'q032':dict(question='How does decomposing RAG chunks into atomic statements and generating questions over those statements affect dense-retrieval recall compared with the original chunks?',reference_answer='Retrieval using atomic statements has higher chunk recall than retrieval using the original chunks. Matching user queries to synthetic questions generated over those atoms, then retrieving the associated chunks, gives a further recall improvement.',reason='Câu cũ cho phép trả lời bằng query rewriting hoặc các phương pháp retrieval khác. Bản sửa hỏi đúng so sánh atoms/chunks/synthetic questions được abstract nêu.'),
 'q033':dict(question='In REALM single-sample test-time adaptation, what problems can result from discarding unreliable inputs, and what ideas motivate the proposed robustness framework?',reference_answer='Discarding unreliable samples can bias updates, slow adaptation and harm generalization. REALM introduces a framework inspired by self-paced learning and robust loss functions to improve robustness to noisy samples.',reason='Chỉ hỏi các hệ quả và nền tảng được abstract nêu; không yêu cầu công thức hoặc khẳng định down-weight chưa có trong nguồn.'),
 'q035':dict(question='How do kernel scores support uncertainty estimation using generated samples alone across image, audio and language, and what evidence is reported on CoQA and TriviaQA?',reason='Thay cụm framework không xác định bằng kernel scores và hai bộ đánh giá cụ thể, phân biệt với các framework uncertainty khác.'),
 'q037':dict(question='In LLM-driven objective discovery for offline preference optimization, how are candidate losses proposed, and which two loss forms does DiscoPOP combine?',reference_answer='An LLM is iteratively prompted to propose and implement new preference-optimization losses using previously evaluated performance metrics. The discovered DiscoPOP objective adaptively blends logistic and exponential losses.',reason='Giới hạn vào quy trình khám phá loss và cấu trúc DiscoPOP, thay vì hỏi mọi cách tìm objective để alignment.'),
 'q038':dict(question='Without large pretraining datasets, how do neural memory and MAML-trained hypernetworks support adaptation, and which 3D tasks illustrate reuse of their priors?',reference_answer='Neural memory enables adaptation to non-stationary distributions from a few samples. Hypernetworks trained with MAML learn more generalizable priors than standard networks; they acquire priors from a handful of scenes for faster text-to-3D generation and transfer priors to segment novel 3D scenes with limited data.',reason='Câu cũ là vấn đề few-shot rất rộng. Bản sửa bám tổ hợp neural memory, MAML và hai ứng dụng 3D được nguồn mô tả.'),
 'q047':dict(question='What objective does PILAF response sampling align preference learning with, and what theoretical perspectives and RLHF settings are reported?',reference_answer='PILAF aligns preference learning with maximizing the underlying oracle reward. The abstract reports optimality from optimization and statistical perspectives and strong performance in iterative and online RLHF.',reason='Không suy diễn thủ tục nội suy từ tên phương pháp; hỏi mục tiêu, phạm vi bảo đảm và thiết lập đã nêu.'),
 'q051':dict(question='In AdaFish low-rank fine-tuning, what structure of the generalized Fisher matrix makes second-order updates computationally practical?',reason='Đặt rõ AdaFish và generalized Fisher để phân biệt với các phương pháp low-rank adaptation không dùng lập luận này.'),
 'q052':dict(question='How does RAIS-DR combine image-quality handling with subgroup-fairness assessment, and what does its evaluation against EyeArt report?',reference_answer='RAIS-DR combines preprocessing, quality assessment and three specialized DR classifiers. On 1,046 patients unseen by both systems, it reports improved F1, accuracy and specificity versus EyeArt; Disparate Impact and Equal Opportunity Difference indicate equitable performance across demographic subgroups in that evaluation.',reason='Đặt rõ hệ thống và comparator; giữ kết luận fairness trong phạm vi đoàn hệ đánh giá thay vì coi là bảo đảm chung.'),
}
NOTES={
 's2001':'Phân biệt transductive dùng biểu diễn test không nhãn với việc dùng nhãn test; accuracy/F1 đều 0.86 trong nguồn.',
 's2002':'Đủ chuỗi sentiment retrieval → Emotion Prompt; giữ CCC 0.593 và MAE 3.95 đúng AVEC 2019.',
 's2003':'Đủ ba corpus, dịch sang 10 ngôn ngữ và lọc bằng perplexity; không bổ sung tên ngôn ngữ không có trong abstract.',
 's2004':'11 review = 9 ChatGPT + 2 AI rộng; mốc tìm kiếm đã có nguồn.',
 's2005':'Hai task FHIR được nêu trực tiếp; không suy ra bảo đảm privacy chỉ từ private deployment.',
 's2007':'Tách latency trên simulated goal drift khỏi so sánh false positive với static thresholds.',
 's2008':'Giữ kết luận định tính trong nghiên cứu 10 ngôn ngữ; không bịa accuracy hoặc tên từng ngôn ngữ.',
 's2009':'Ba reward và vai trò text bridge đều được nguồn nêu trực tiếp.',
 's2010':'Cả tăng cooperation và tăng khả năng bị exploitation đều có trong nguồn.',
 's2011':'Giữ r/p và quy kết kết luận cho nghiên cứu; không biến tương quan không rõ thành bằng chứng không có hiệu ứng ở mọi trường hợp.',
 's2012':'45%/67% là mức tăng median silhouette được báo cáo; giữ đúng tên biến thể K-means/agglomerative.',
 's2013':'Giữ Recall@1 từ 7.5% lên 19.8%, không nhầm mức tăng tuyệt đối và tương đối.',
 's2014':'Remediator và value-impact được nguồn nêu; DICE tìm được bằng lexical search không đáp ứng ngữ cảnh business negotiation này.',
 's2015':'Chỉ hỏi hạn chế vector toàn câu và thuộc tính span aggregation; không khẳng định tiết kiệm tính toán định lượng chưa có.',
 's2016':'Đánh giá trong generation và hybridization được nêu trực tiếp; không khẳng định hybrid luôn thắng.',
 's2017':'Nguồn cho phép giải thích metadata thiếu/cũ, review phong phú và RA-Rec state tracking; không thêm metric.',
 's2018':'Ba nguyên tắc human context, vượt nhóm và biến đổi theo thời gian đều có nguồn.',
 's2020':'Giữ điều kiện moderate noise và visual filtering; không gán mức tăng số chưa có.',
 's2021':'Đủ ba nhóm phương pháp, bốn LVLM và kết luận supervised uncertainty trong phạm vi đã thử.',
 's2022':'18.2% là WER improvement tương đối; 54% forgetting giảm so với naive; EWC có trade-off plasticity.',
 's2023':'Phân biệt pretraining/fine-tuning với adaptation in-context không gradient update mới.',
 's2024':'Giữ phân biệt giả thuyết với nguyên nhân đã được chứng minh; freezing experiments inconclusive.',
 's2025':'Precision cao không đủ cho F1 khi recall 12.87%; ba classifier đều có nguồn.',
 's2026':'Browser/TSV và bốn task được nêu; không suy ra mọi bài toán annotation được hỗ trợ.',
 's2027':'Phân biệt accuracy-oriented calibration và calibration theo human distributions trên BEiT3.',
 's2028':'Giữ up to cho 2.16x latency và 3.18x throughput; ký hiệu 3.18x không phải con số ngoài nguồn.',
 's2029':'Chỉ giữ role optimizer và mức chứng cứ preliminary; không tự thêm kết luận định lượng hoặc bảo đảm tổng quát.',
 's2030':'Explicitness và narrative element là hai thuộc tính khác nhau; chứng cứ được mô tả là preliminary.',
 's2031':'0.5% parameters, adaptive instruction MoE và mutual-information linkage đều trong nguồn; không thêm công thức.',
 's2032':'Giữ hai mức budget/cache và tỷ lệ 1.7%/85% trong LongBench; không lẫn thử nghiệm Needle.',
 's2033':'9–52% áp dụng related probe score của các auxiliary tasks, không phải toàn bộ recommendation accuracy.',
 's2034':'Hai dataset, không fine-tuning/ground-truth rationales; không bịa tên hai dataset.',
 's2035':'Hai chiều populist language và host ideologies đều có nguồn; không đưa diễn giải chính trị ngoài bài.',
 's2036':'Nguồn có metadataset, năm model/sáu dataset; không khẳng định đã public release hoặc tự đặt tên model.',
 's2037':'Can thiệp đích và nhiễu ngoài đích được phân biệt; không coi amnesic probing là chứng minh nhân quả tuyệt đối.',
 's2038':'Adversarial consistency không teacher và một/hai sampling steps được nguồn nêu.',
 's2039':'Không longer-context training; gains có cả trong/vượt context với cùng lượng knowledge; không suy diễn orchestration cụ thể.',
 's2040':'Drift các functional directions và gradient orthogonality có nguồn; không hứa triệt tiêu hoàn toàn forgetting.',
 's2041':'Chỉ hỏi phép đo và human evaluation được nêu, không hỏi period/term/model ranking chưa liệt kê.',
 's2042':'Context-size curriculum khác ba curricula của CLIMB; không coi bài CLIMB là gold thay thế cho kết quả này.',
 's2043':'Giới hạn vào roles, hidden states/output features và semantic similarity; không hỏi công thức constraint không có.',
 's2044':'Mixture thuộc tokenizer training set, không tự gán thành LM pretraining distribution.',
 's2045':'Giữ up to 78.52% và fluency drop 0.052%; không hứa vô hại hoặc loại mọi toxicity.',
}
ISSUES={
 's2006':dict(issue='Abstract mô tả gradient manifolds và dynamic adjustment rất chung, không định nghĩa thao tác đủ rõ để chấm đáp án cơ chế. Có thể thu hẹp thành câu hỏi input/output của mạng phụ, nhưng sẽ khá dễ.',recommendation='Thay bằng câu có thao tác và bằng chứng rõ hơn.'),
 's2019':dict(issue='Nguồn chỉ nói một số paradigm mạnh về question, số khác mạnh về question-context, không nêu paradigm nào hay kết quả cụ thể. Đáp án hiện tại đúng ở mức tóm tắt nhưng quá chung để phân biệt chất lượng hệ thống.',recommendation='Thay; nếu giữ thì chỉ dùng như câu tổng quan định tính, không chấm chi tiết tên paradigm.'),
 'q056':dict(issue='Câu gốc đòi chiến lược post-training/inference nhưng abstract chỉ mô tả phạm vi survey. Bản sửa về phạm vi trả lời được, nhưng vẫn gần như một câu giới thiệu survey, ít giá trị đo chất lượng.',recommendation='Thay bằng câu có phương pháp cụ thể; bản cũ vẫn được giữ nguyên trong QA gốc.'),
}
REPLACEMENTS=[
 dict(target_id='s2006',paper_id='2412.07675',question='How does RAZOR define the shortcut space used for text rewriting, and does it require prior knowledge of dataset biases?',reference_answer='RAZOR uses token statistics and positional information to define the shortcut space, then uses LLMs to iteratively rewrite potentially biased segments with heuristically selected alternatives. It does not require prior bias information.',evidence=['iteratively rewrite potentially biased text segments by replacing them with heuristically selected alternatives in a shortcut space defined by token statistics and positional information','without requiring prior bias information']),
 dict(target_id='s2019',paper_id='2509.26314',question='How does LTO obtain a reward signal directly from Huginn-3.5B latent thoughts, and what process does that signal optimize?',reference_answer='A latent classifier predicts answer correctness directly from latent thoughts. LTO uses that classifier as a Latent Reward Model in a probabilistic algorithm to optimize the latent thinking process.',evidence=['a latent classifier can reliably predict answer correctness directly from latent thoughts','Latent Thinking Optimization (LTO), a probabilistic algorithm that employs the latent classifier as a Latent Reward Model (LRM) to optimize the latent thinking processes']),
 dict(target_id='q056',paper_id='2402.10805',question='In Generative Cross-Modal Retrieval that memorizes images inside an MLLM, how are target images represented and how do the two training stages differ?',reference_answer='Images are represented by unique identifier strings. The first stage teaches the multimodal model image-identifier associations; the second teaches it to generate the target image identifier from a text query.',evidence=['assigns unique identifier strings to represent images and involves two training steps: learning to memorize and learning to retrieve','The first step focuses on training the MLLM to memorize the association between images and their respective identifiers','The latter step teaches the MLLM to generate the corresponding identifier of the target image, given the textual query input']),
]

def main():
    snapshot=BASE/'content_review_v1.json'
    if snapshot.exists():
        current={r['id']:r for r in load(BASE/'qa_candidates.jsonl')}
        audit=json.loads(snapshot.read_text(encoding='utf-8'))
        assert all(fingerprint(current[r['id']])==r['source_fingerprint'] for r in audit['rows']),'Dataset changed: perform a new content review, do not reuse this snapshot'
        print('Existing content review still matches. No changes.');return
    backup=ROOT/'outputs/benchmark-content-review-v1';backup.mkdir(parents=True,exist_ok=True)
    changes=[]
    for path in sorted(BASE.glob('authored_single_*.json')):
        before=path.read_bytes();rows=json.loads(before)
        for row in rows:
            if row['id'] in NEW_EDITS:
                changes.append(dict(id=row['id'],before={k:row[k] for k in NEW_EDITS[row['id']]},after=NEW_EDITS[row['id']]))
                row.update(NEW_EDITS[row['id']])
            if row['id']=='s2015':
                row['evidence'].append('a dataset based on the STS-B dataset with additional generated text')
        if not (backup/path.name).exists():(backup/path.name).write_bytes(before)
        dump(path,rows)
    path=BASE/'legacy_revision_proposals.json';before=path.read_bytes();proposals={r['id']:r for r in json.loads(before)}
    if not (backup/path.name).exists():(backup/path.name).write_bytes(before)
    for qid,edit in OLD_EDITS.items():
        changes.append(dict(id=qid,after=edit));proposals.setdefault(qid,dict(id=qid)).update(edit)
    dump(path,sorted(proposals.values(),key=lambda r:r['id']))
    compile_draft()
    rows=load(BASE/'qa_candidates.jsonl');audit_rows=[]
    for row in rows:
        if row['type']!='single':continue
        qid=row['id'];issue=ISSUES.get(qid)
        note=(issue['issue'] if issue else NOTES.get(qid,proposals.get(qid,{}).get('reason','Đối chiếu nội dung với title/abstract ở lượt rà nguồn trước; không thấy thiếu sót trọng yếu cần đổi đáp án. Không xác nhận đây là lượt gán nhãn độc lập của người.')))
        audit_rows.append(dict(id=qid,status='needs_hill_decision' if issue else 'cleared',note=note,source_fingerprint=fingerprint(row)))
    assert len(audit_rows)==105
    audit=dict(reviewed_at=datetime.now(timezone(timedelta(hours=7))).isoformat(),reviewer='Codex',reviewer_kind='assistant_source_review',
               scope='Title/abstract grounded content review, not independent human annotation or full-paper scientific verification. Prior bridge approval unchanged.',
               ambiguity_check='Exact question duplicates checked. BM25 top alternatives inspected for 16 non-rank-1 single questions; rank is only an inspection aid, never proof of ambiguity, uniqueness or a difficulty selection rule. Candidate ranks were computed before revisions.',
               reviewed_single=105,cleared_single=102,needs_hill=list(ISSUES),bridge_preserved=45,rows=audit_rows)
    corpus={p['paper_id']:p for p in load(ROOT/'data/processed/benchmark_v2_corpus.jsonl')}
    occupied={pid for row in rows for pid in row['gold_ids']}
    cases=[]
    for proposal in REPLACEMENTS:
        assert proposal['paper_id'] not in occupied
        p=corpus[proposal['paper_id']];text=p['title']+'. '+p['abstract'];spans=[]
        for quote in proposal['evidence']:
            span=locate_evidence(text,quote);assert span,(proposal['target_id'],quote)
            spans.append(dict(text=span.text,start=span.start,end=span.end))
        current=next(r for r in rows if r['id']==proposal['target_id'])
        cases.append(dict(id=proposal['target_id'],**ISSUES[proposal['target_id']],current_question=current['question'],current_answer=current['reference_answer'],
                          replacement=dict(**proposal,title=p['title'],abstract=p['abstract'],evidence_spans=spans),applied=False))
    dump(BASE/'hill_decisions_needed.json',cases);dump(backup/'changes.json',changes);dump(snapshot,audit)
    print(json.dumps(dict(content_cleared=147,needs_hill=list(ISSUES),replacement_proposals_validated=len(cases),human_decisions_written=0),indent=2))
if __name__=='__main__':main()
