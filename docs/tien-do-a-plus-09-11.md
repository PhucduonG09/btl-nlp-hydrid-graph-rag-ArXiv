# Tiến độ đối chiếu tiêu chí A+ — 09/11/2026

Nguồn định hướng: `D:\Study\NLP\Tổng quan & tiêu chí A+.pdf`, tài liệu Hill cung cấp, chưa xác nhận là rubric chính thức. Chi tiết lịch sử trước generation được giữ ở `tien-do-a-plus-before-generation.md`.

## Đã hoàn thành và kiểm chứng

- [x] Benchmark **v2 assisted**: 150 câu = 105 single + 45 bridge, corpus 20.500 paper; 122 DEV / 28 test không chung gold paper. Manifest khóa hash dữ liệu. Không gán 102 lượt rà nguồn của trợ lý thành annotation người.
- [x] Giữ nguyên ba workbook, cấu trúc và nội dung. Audit hash cả ba đều đạt.
- [x] Baseline BM25/BGE/RRF, mở rộng văn bản, graph k=1/2/3/PPR/fusion: 43 cấu hình trên DEV. Config đã khóa trong `results/benchmark_v2_assisted/dev_01/selected_config.json`.
- [x] Đồ thị MENTIONS toàn corpus: 33.643 entity / 58.026 cạnh. Xây từ corpus, không đọc QA/gold; không gọi nó là KG USES/PROPOSES.
- [x] Extractor rule độc lập, scorer matching một-một và ablation typed graph toàn corpus. Rule typed F1 = 0,124 trên gold 60 paper; quy tắc yếu và khác phạm vi extractor LLM.
- [x] Reranker MiniLM có revision/hash, áp dụng trên cùng pool top-50. Trên bridge DEV, CE thuần làm Graph All@10 giảm từ 26/31 xuống 8/31; cấu hình Graph cuối giữ nguyên, báo cáo cả kết quả âm.
- [x] Generator dùng cùng prompt, top-10 nguồn, context cap 6.000 token proxy, output tối đa 768, temperature 0. Claude Sonnet 5 sinh đủ 244 đáp án; GPT-6-sol chấm đủ 244. Một lỗi JSON judge đã được chạy lại; lịch sử lỗi và token vẫn giữ.
- [x] Kiểm tra model trả về: pilot Sonnet 5.5 có trộn nhãn, tách khỏi kết quả chính. Lượt chính 244 đáp án cùng nhãn claude-sonnet-5, judge cùng nhãn gpt-6-sol. Nhãn provider không chứng minh trọng số thực tế.
- [x] A-only diagnostic 31 bridge DEV: 24 từ chối / 7 một phần / 0 đúng đầy đủ theo judge; không thay gold hoặc cấu hình theo kết quả này.
- [x] Gói 50 đáp án chấm người, 25 single / 25 bridge và 25 RRF / 25 Graph, 50 câu khác nhau. Ẩn hệ thống và điểm AI, lưu nhãn người riêng. Hiện 0/50 được người chấm.
- [x] Demo hỏi đáp thật, trích dẫn nguồn và đường nối; giao diện kết quả DEV và tiến độ. Kiểm tra lời gọi thật với câu DEV; không dùng câu test.
- [x] Nhật ký token theo model và mọi lượt thử, bao gồm pilot loại khỏi phân tích và retry; chưa có đơn giá nên không bịa chi phí USD/VND.
- [x] Bộ kiểm thử hiện tại: 59 bài đạt, gồm giao diện chấm 50 mẫu và trang nhận xét trợ lý không tự ghi nhãn; xử lý bất đồng/uncertain/reference lỗi giữ pending, cổng kiểm tra test và thống kê hiệu chuẩn. Đợt generation/judge/probe/A-only/demo ghi nhận 3.669.882 token input + 319.505 output; chưa gồm extraction lịch sử. Có 19 đáp án chứa tổng 20 witness quote cần rà riêng.

## Kết quả DEV hiện tại

| Chỉ số | RRF | Graph MENTIONS |
|---|---:|---:|
| Single có gold trong top-10 | 90/91 | 90/91 |
| Bridge đủ hai gold trong top-10 | 2/31 | 26/31 |
| Single đúng đầy đủ, judge tự động | 59/91 | 70/91 |
| Bridge đúng đầy đủ, judge tự động | 3/31 | 23/31 |
| Đúng đầy đủ toàn DEV, judge tự động | 62/122 | 93/122 |

Config chọn trên DEV; số generation **chưa hiệu chuẩn người**, chưa phải test. Benchmark bridge được lọc theo BM25 nên không suy rộng cho mọi truy vấn. Graph giúp lấy paper còn thiếu trong lát cắt này; chưa chứng minh KG ngữ nghĩa tốt hoặc mọi lỗi đều là hallucination. Single mới: RRF 27/32 đúng, Graph 26/32; cải thiện single tổng thể tập trung ở nhóm câu cũ.

Faithfulness được tính trên đáp án có khẳng định thực tế và không bị chương trình/judge đánh dấu từ chối. Không gán từ chối = 1. Cú pháp trích dẫn hợp lệ không chứng minh nội dung đúng. Có 20 claim judge đánh dấu supported nhưng bằng chứng trích nguyên văn chưa qua kiểm tra; lưu riêng để rà, không tự gọi đó là 20 hallucination.

Báo cáo chi tiết: `results/benchmark_v2_assisted/dev_01/generation_prepared/claude5_controlled/README.md`, `evaluation_summary.json`, `answers_for_audit.csv`, `judge_witness_review.json`, `token_ledger.json`.

## Việc tiếp theo

Đã rà riêng 50 đáp án theo nguồn bởi trợ lý (không ghi thành nhãn người): 33 đáp ứng ý chính, 13 từ chối, 3 một phần, 1 lạc đích. Đối chiếu 91 đoạn nguồn; giữ nguyên outputs/gold/Excel. Trong 25 bridge của gói có 25 A trong context, 10 B có và 15 B thiếu. H036/bd002 có rủi ro không xác định duy nhất paper đánh giá GPT-4V, cần Hill chốt ở erratum/phiên bản kế tiếp; không tự sửa benchmark frozen. Trang nhận xét `http://127.0.0.1:8503/?view=assistant`, dữ liệu `data/eval/generation_calibration_v2/assistant_review_v1/`.

Cập nhật kỹ thuật: đã thêm protocol test cố định và runner retrieval có resume, kiểm tra hash, kiểm tra đủ 50 nhãn người trước lần mở QA test. Engine triển khai khớp top-100 của cả năm phương pháp trên sáu câu DEV. Đã thêm so sánh hỗ trợ nguồn vào báo cáo hiệu chuẩn (bên cạnh correctness); khi chưa có nhãn người, agreement/kappa/F1 đều để null. Audit 20 witness quote: 4 quote ghép bằng dấu ba chấm, 14 quote không nguyên văn cần đối chiếu nguồn, 2 phát biểu thiếu thông tin trong đáp án từ chối không có positive witness. Không sửa kết quả judge gốc hoặc tự đổi các cảnh báo này thành hallucination. Chi tiết ở `docs/protocol-test-v2.md` và `judge_witness_diagnostics.json`.

1. **Nhóm chấm 50 mẫu** tại http://127.0.0.1:8503: tên người chấm → đọc câu/đáp án/nguồn → chấm mức đúng và hỗ trợ nguồn → lưu. Các trường hợp chưa chắc, bất đồng hoặc reference lỗi giữ pending; không nhờ AI điền thay nhãn người. Chấm xong chạy `scripts/summarize_generation_v2.py` để tính agreement/kappa/macro-F1 và báo pending.
2. Rà calibration và chốt protocol judge/generator trước khi đánh giá **28 test** một lần theo config đã khóa. Không chạy grid tuning trên test, không sửa dataset để tăng điểm.
3. Xác nhận provenance độc lập của gold extraction/annotator A/B, hòa giải bất đồng. Agreement mục hiện chỉ 0,471; kappa relation 0,930 là trên các cặp tên đã khớp, không phải toàn bộ annotation.
4. Nếu tiếp tục KG ngữ nghĩa dùng LLM: chọn lát cắt corpus/topic trước, ghi model/schema và ngân sách token, không chọn theo gold test. KG 555 paper mở rộng theo gold DEV chỉ là diagnostic. Kết quả MENTIONS/rule không thay thế thực nghiệm semantic LLM công bằng.
5. Hoàn thiện báo cáo/slides từ bảng thật, thống kê held-out có CI và hạn chế; điền tên nhóm/GV/ngày thực tế. Kiểm tra môi trường cài mới và provenance embedding cache cũ (revision lúc tạo cache ban đầu chưa ghi).

Hill đã cho phép tiếp tục gọi API và sẽ bổ sung token khi hết. Không chờ đơn giá để thực hiện phần đã cho phép; giữ số token thật và bổ sung bảng giá khi Hill cung cấp. Không sửa `.env` hoặc hiển thị khóa.

**Chưa hoàn tất toàn đề tài/A+.** Các việc cần người gán nhãn, test chưa chạy, provenance gold, semantic KG công bằng và báo cáo cuối không được tự tick.
