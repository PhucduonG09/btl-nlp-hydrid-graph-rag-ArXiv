# Quy trình test v2 cố định

Mục tiêu: đánh giá 28 câu test đã tách theo gold paper, sau khi đã chấm người gói 50 đáp án DEV. Không dùng test để lựa chọn cấu hình. File protocol máy đọc được đặt tại `results/benchmark_v2_assisted/test_01/protocol.json`.

## Phạm vi hiện đã triển khai

1. `scripts/run_locked_test_v2.py prepare`: khóa cấu hình và hash mã/corpus/graph/embedding; lấy SHA test từ manifest, **không mở QA test** và không gọi API.
2. `scripts/run_locked_test_v2.py check-dev`: so engine triển khai với kết quả top-100 đã lưu trên sáu câu DEV, cả năm phương pháp. Đây là kiểm tra nhất quán, không tuning.
3. `scripts/run_locked_test_v2.py execute`: chỉ sau khi 50 mẫu có nhãn người đã chốt. Kiểm tra hash và engine, ghi biên nhận trước lần mở test, chạy retrieval cố định và chuẩn bị 56 bộ nguồn generation. **Lệnh này không gọi LLM**. Nếu tiến trình bị ngắt, chạy lại sẽ tiếp tục các câu chưa xử lý với protocol không đổi; nếu hoàn tất, từ chối chạy mới.

Lệnh đầy đủ cần tiền tố `.venv-kg/Scripts/python.exe -X utf8` tại thư mục gốc repo. Chuẩn bị protocol chưa có nghĩa test đã được đánh giá. Trạng thái hoàn tất được xác định bằng `execution_complete.json`, không suy ra từ sự tồn tại thư mục `test_01`.

## So sánh đặt trước

- Năm phương pháp: BM25, dense BGE, RRF, mở rộng văn bản từ cùng ba linked seed, Graph MENTIONS + RRF.
- So sánh chính: Graph với RRF, chỉ số Bridge All@10. Báo cáo Single R@10, MRR@10 và từng hop riêng. Luôn giữ đầy đủ năm phương pháp, không chỉ chọn bảng có lợi.
- BM25 k1=1,5/b=0,75; BGE snapshot đã ghi; RRF baseline k=10. Graph query-linking ba seed, một bước Paper→Entity←Paper, fusion k=60 với trọng số baseline:graph = 1:2. Đối chứng mở rộng văn bản dùng đúng seed/corpus/BM25.
- Không đưa gold, kiểu câu, đáp án tham chiếu vào truy xuất hoặc prompt generator. Gold chỉ dùng tính chỉ số sau truy xuất. MENTIONS không được trình bày như USES/PROPOSES.
- Embedding cache phải giữ đúng SHA của DEV; provenance revision lúc tạo cache ban đầu chưa xác minh độc lập. Không che hạn chế này bằng revision encoder hiện tại.

## Phần generation/statistics cần hoàn tất trước khi công bố test

Sau chấm người, đọc agreement correctness và hỗ trợ nguồn, rà các reference bị đánh dấu; 50 nhãn hoàn tất không tự chứng minh judge đáng tin. Chốt judge/prompt trước lượt generation test. Runner retrieval đã chuẩn bị nguồn/prompt giống DEV (top-10, cap 6.000 proxy token, max output dự kiến 768, temperature 0), nhưng **chưa thực hiện hoặc tuyên bố có 56 đáp án test**.

Cần tiếp tục runner API test có resume/nhật ký mọi lượt thử và hash đầu vào, rồi chấm với protocol đã chốt. Báo cáo CI 95% theo thành phần gold paper, so sánh theo cặp và số mẫu nhỏ. Không biến khoảng tin cậy DEV đã chọn tham số thành bằng chứng confirmatory. Không thay câu hỏi hoặc retune sau khi thấy test. Nếu phát hiện lỗi dữ liệu thật, ghi erratum và báo cả phạm vi loại mẫu.

## Kiểm tra offline trên máy

`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_experiment_v2.ps1 -CheckEngine`

Lệnh kiểm tra hash workbook, tổng hợp lại báo cáo/hiệu chuẩn, audit quote, so engine DEV và chạy unit/UI tests. Không sinh lại dữ liệu và không gọi API. Các test chấm người dùng fixture trong thư mục tạm, không ghi SQLite chấm thật.

## Amendment A1 (2026-11-11) - `reference_issue` là cờ phân tầng, không còn chặn gate

**Bối cảnh.** Sau khi hai người chấm (Phúc, duyanh) hoàn tất 50 mẫu, 19/50 mẫu có ít nhất một cờ `reference_issue` và 2 mẫu bất đồng nhãn. Phần lớn cờ (15 mẫu) thuộc câu hỏi "mô tả tài nguyên X" mà đáp án tham chiếu lấy từ paper gốc không nằm trong 10 abstract được truy xuất, nên model từ chối là đúng còn tham chiếu không suy ra được từ nguồn. Luật cũ (`resolved_ratings`) chặn mọi mẫu có cờ này.

**Thay đổi.** Thêm `resolved_ratings_v2`: một mẫu *resolved* khi mọi người chấm cho cùng `(quality, support)` và không có `uncertain`. `reference_issue` được giữ trong kết quả như cờ phân tầng. Bất đồng nhãn vẫn *pending*, không bao giờ được chốt theo người ghi sau. `run_locked_test_v2.py` và `summarize_generation_v2.py` dùng hàm mới; hàm cũ giữ nguyên để đối chiếu.

**Trung thực về thời điểm.** Đây là chỉnh sửa hậu nghiệm, thực hiện sau khi đã thấy nhãn người chấm nhưng **trước khi mở `qa_test`**; nhãn không được dùng để chọn cấu hình truy xuất hay generation. Không có nhãn nào do AI điền. Báo cáo phải nêu điều chỉnh này.

**Bắt buộc khi báo cáo.**
1. Báo agreement/kappa giữa hai người chấm trên nhãn gốc (trước thảo luận) và trên nhãn sau thảo luận.
2. Tách tầng `reference_unsupported` (n được ghi trong `human_calibration_status.json`, mục `reference_issue_strata`) khi đánh giá judge.
3. Khi chấm generation trên test, báo riêng tỷ lệ từ chối và chất lượng sinh có điều kiện gold đã nằm trong ngữ cảnh; không tính từ chối đúng là sai.
4. `code_sha256` trong `protocol.json` có `src/eval/human_calibration.py`; sau thay đổi này phải chạy lại `prepare` và `check-dev` trước `execute`.

## Amendment A2 (2026-11-11) - chỉ số generation trên test, đặt trước khi mở lượt sinh

**Căn cứ (DEV, 50 mẫu người chấm sau A1).** Judge so với người: `quality` agreement 76%, κ 0,58; `support` agreement 82%, κ 0,70. Mọi lệch đều theo một hướng (judge khắt khe hơn): 10/35 câu người coi `correct` bị judge hạ thành `partial`; judge ghi `partial` 13 lần mà người chỉ đồng ý 1 lần. Nhãn người có 35 correct, 14 abstained, 1 partial, 0 incorrect nên judge **chưa được kiểm chứng** cho `partial` và `incorrect`; macro-F1 bốn lớp không có ý nghĩa và không được báo cáo. Ở tầng `reference_supported` (n=34) κ=0 vì người chấm cho toàn bộ là `correct` (suy biến); chỉ báo agreement.

**Quy định cho lượt generation test (không đổi judge/prompt, không chạy lại DEV để tuning):**
1. Chỉ số chính của generation: (a) tỷ lệ từ chối; (b) tỷ lệ trả lời thực chất = `correct` + `partial`, báo kèm điều kiện gold đã nằm trong ngữ cảnh; (c) faithfulness/support.
2. `correct_rate` (tách `correct` khỏi `partial`) chỉ báo như độ nhạy của judge, kèm bảng ma trận nhầm lẫn DEV; **không rút kết luận** về hệ thống nào sinh tốt hơn từ ranh giới này.
3. Mọi so sánh Graph và RRF ở bước sinh phải nêu rõ phần chênh lệch do retrieval (gold trong ngữ cảnh) và không gán cho bước sinh.
4. Từ chối khi gold không nằm trong ngữ cảnh không tính là sai.

**Trung thực về thời điểm.** Quy định này đặt sau khi thấy kết quả calibration trên DEV, trước khi mở `qa_test` và trước mọi lượt sinh trên test. Báo cáo phải nêu điều này.
