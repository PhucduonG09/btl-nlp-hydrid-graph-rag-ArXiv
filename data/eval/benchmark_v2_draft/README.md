# Benchmark v2 — bản nháp để duyệt

Đã ghép 150 ứng viên. Sau chỉ dẫn “thay cả 3” của Hill, bộ hiện có **59 single cũ + 46 single mới/thay thế + 45 bridge giữ nguyên**. Không còn trường hợp mơ hồ chờ chốt ở lượt rà nguồn này. Chưa đóng băng làm test hoặc tự xác nhận đủ tiêu chí gán nhãn người độc lập.

- `qa_candidates.jsonl`: bản ghép làm việc, tách biệt hoàn toàn với QA cũ và bridge v1.
- `new_single_drafts.jsonl`: 46 single mới/thay thế, có 123 đoạn bằng chứng với offset được kiểm tra khớp nguồn.
- `legacy_revision_proposals.json`: 16 bản sửa câu cũ trong bản làm việc v2, sau lượt rà nội dung; nội dung QA gốc được giữ nguyên.
- `content_review_v1.json`: giữ nguyên lịch sử lượt rà ban đầu, có 3 trường hợp cần Hill chốt.
- `content_review_v2.json`: trạng thái hiện tại sau khi thay cả 3; không còn câu chờ chốt nội dung. Lượt rà nguồn của trợ lý không được ghi thành gán nhãn người độc lập.
- `accepted_replacements.json`: ghi nhận đúng chỉ dẫn chat của Hill cho s2006, s2019 và q056, cùng bản trước, paper thay thế và bằng chứng.
- `hill_decisions_needed.json`: giữ lịch sử đề xuất; cả 3 đã có `applied=true`.
- `legacy_single_sources.jsonl`: 60 câu cũ kèm paper nguồn. Nội dung đã được đối chiếu với tiêu đề/abstract; không tự gắn lại nhãn người duyệt.
- `new_single_sources.jsonl`: 45 paper chọn bằng quy tắc cố định, thuộc cs.CL, không trùng gold của QA cũ/bridge; không dùng KG hay kết quả retrieval để chọn.
- `legacy_comparisons_separate.jsonl`: 10 câu so sánh cũ, không nhập thành bridge.
- `validation.json`: kiểm tra cơ học và các con số cần đối chiếu. Hai cờ q004/s2028 là cảnh báo do định dạng số trong LaTeX hoặc ký hiệu đơn vị; nguồn có số 50 Hz và 3.18x tương ứng, không phải lỗi đáp án đã được chứng minh.

## Cách duyệt trên máy

Từ gốc repo:

```powershell
.venv-kg/Scripts/python.exe -m streamlit run review_benchmark.py --server.address=127.0.0.1 --server.port=8502
```

Giao diện hiện mặc định chọn nhóm **Các câu đã thay**, hiển thị s2006 (RAZOR), s2019 (LTO) và q056 (Generative Cross-Modal Retrieval). Paper và bằng chứng trên màn hình là bản mới. Không yêu cầu Hill rà lại toàn bộ 105 single. Nếu phát sinh sửa nội dung/nguồn làm mất hiệu lực lần rà trước, nhóm **Cần Hill chốt** sẽ xuất hiện lại.

Nhật ký người duyệt vẫn được lưu riêng để dùng khi cần đáp ứng tiêu chí gán nhãn người. Nếu nhóm đã duyệt các câu cũ giữ nguyên trước đây, có thể ghi nhận provenance đó trong nhóm **Toàn bộ câu single cũ**. Không tự gọi lượt rà của trợ lý hoặc việc xác nhận lượt duyệt cũ là một lần gán nhãn độc lập mới.

Các quyết định được ghi vào `review_decisions.sqlite3` trong thư mục này. Không sửa file Excel, `qa.jsonl` hay bridge v1. Khi nội dung nguồn của một câu thay đổi, quyết định cũ của câu đó hết hiệu lực để tránh dùng nhầm phê duyệt cho một bản khác.

45 bridge giữ nguyên và chỉ xem đọc. Bản nguồn `qa_candidates.jsonl` được compiler tạo lại; chỉnh sửa của người duyệt nằm riêng trong database nên không bị ghi đè khi biên dịch lại. Chưa có test split v2 hay kết quả test mới. Các câu cũ đã có lịch sử dùng trong thực nghiệm, không được gọi là hoàn toàn chưa từng thấy.

## Tạo lại dữ liệu làm việc

```powershell
.venv-kg/Scripts/python.exe -X utf8 scripts/prepare_benchmark_v2.py
.venv-kg/Scripts/python.exe -X utf8 scripts/compile_benchmark_v2_draft.py
```

Kiểm tra mức độ sẵn sàng sau khi lưu các quyết định:

```powershell
.venv-kg/Scripts/python.exe -X utf8 scripts/finalize_benchmark_v2.py
```

Lệnh hiện kiểm tra nhật ký duyệt người, tách biệt với trạng thái rà nội dung của trợ lý. Vì vậy số pending trong lệnh này chưa phải số câu mơ hồ cần Hill chốt. Không tự ghi nhãn người để làm số liệu đẹp. Chỉ khi đủ điều kiện của chế độ xuất bản tương ứng mới đóng băng; bước xuất bản phải công khai tỷ lệ gán nhãn người/trợ lý. Split cần giữ bridge v1, gom nhóm theo gold paper, và ghi lịch sử phơi lộ của câu cũ. Không ghi đè phiên bản đã đóng băng.

Corpus v2 dự kiến có 20.500 paper ở `data/processed/benchmark_v2_corpus.jsonl`; corpus v1 20.091 paper không bị thay đổi. Mọi câu bị loại cần được bù trước khi chốt mốc ≥150 câu hợp lệ.
