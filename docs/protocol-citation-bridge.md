# Giao thức: bộ bridge dựa trên trích dẫn tự nhiên (khai báo TRƯỚC khi chạy hệ thống nào)

Mục đích: kiểm tra khái quát hóa ngoài bridge dựng từ tên riêng chung (vòng tròn theo thiết kế). Nguồn liên kết là trích dẫn thật (Semantic Scholar), độc lập với từ ngữ.

## Chọn cặp (`scripts/select_citation_pairs.py`, seed 20261112)
A trích dẫn B, cả hai trong corpus v2; A mới hơn B; B có in-degree trong corpus <= 10 (loại tham chiếu nền); mỗi paper tối đa một lần; loại cặp đã nằm trong qa_dev và bridge khó; 320 cặp ngẫu nhiên. Không lọc theo từ vựng, BM25, dense hay graph. Hash lựa chọn: `data/eval/citation_bridge/selection_meta.json`.

## Sinh câu hỏi (`scripts/draft_citation_bridge.py`, LLM do user chạy)
LLM thấy hai abstract và viết một câu hỏi về A mà đáp án chỉ có ở B, không nêu tên B. Bộ lọc tự động chỉ gồm: LLM phán hợp lệ; câu không trả lời được chỉ từ abstract A; câu không chứa tên riêng/viết tắt trong tiêu đề B. B_rank (BM25) chỉ ghi lại làm biến phụ.

## Duyệt người (`make_citation_review_sheet.py`, `finalize_citation_bridge.py`)
Hai người duyệt ĐỘC LẬP, mù (không thấy phán quyết LLM, hạng BM25). Nhận câu khi cả hai chấm hợp_lệ=Y và A_tự_trả_lời=N. Bất đồng được nhóm họp quyết và ghi lại, không tự phân xử. Báo cáo kappa. Bộ được đóng băng bằng hash (`frozen.json`) trước khi chạy hệ thống.

## Hệ thống và chỉ số (đóng băng)
Năm phương pháp cố định như test v2: BM25, dense, RRF, text_expansion, Graph MENTIONS; cấu hình khóa, không tune, không thêm biến thể (CITES và two-hop đã được thử riêng và không được đưa vào). Chỉ số chính: All@10 (cả A và B trong top-10). Phụ: Hop1@10, Hop2@10. So sánh chính: Graph vs RRF (McNemar chính xác hai phía). Phụ: Graph vs dense, text_expansion vs RRF. Báo cáo theo tầng (mô tả, không suy diễn): B_rank BM25 <=10 hay >10; cặp có/không chung thực thể MENTIONS.

## Dự đoán trước (chưa biết kết quả)
Trên trích dẫn tự nhiên, chỉ khoảng 11% cặp chung thực thể (đo ở mức cặp), nên Graph MENTIONS không còn lợi thế lớn như ở bridge tên riêng; kỳ vọng chênh lệch Graph so với RRF nhỏ hoặc không có ý nghĩa, và có thể âm. Nếu Graph vẫn thắng rõ thì đó là bằng chứng mạnh hơn cho giá trị cấu trúc.

## Giới hạn đã biết
Bước LLM "hợp lệ" ưu tiên cặp có nội dung B rõ trong abstract nên có thiên lệch chọn; n sau duyệt có thể chỉ 40-70 nên khoảng tin cậy rộng (đủ phát hiện chênh lệch lớn hơn khoảng 15 điểm); abstract-level, không full-text; trích dẫn không đồng nghĩa với "cần thông tin của B"; mọi kết quả là bổ sung, test v2 vẫn là kết quả confirmatory chính.

## Sửa đổi 1 (2026-11-12, SAU lần chạy thử 5 cặp, TRƯỚC mọi lần chạy đầy đủ và mọi đánh giá hệ thống)
Lý do: trong 4 câu thử, một câu không mô tả A (A_rank 2538, thực chất là câu một bước về B) và ba câu mô tả B bằng từ khóa riêng nên BM25 tìm ngay B (B_rank 1-2). Thay đổi:
1. Prompt sinh câu: câu hỏi phải nhận ra được A bằng nội dung của A, và không được mô tả nội dung riêng của B (chỉ nêu vai trò của B so với A).
2. Điều kiện hợp lệ của bridge: A_rank (BM25) <= 10, tức câu hỏi nhận ra được A. Đây là điều kiện về hop 1, không dùng thông tin B.
3. Phân tích chính chuyển sang tầng B_rank (BM25) > 10, tức B không tìm được trực tiếp từ câu hỏi. Đây là điều kiện giống MuSiQue/bộ v2: chỉ kiểm tra giá trị multi-hop khi hop 2 không tầm thường. Tầng B_rank <= 10 và toàn bộ vẫn báo cáo. Nếu n của tầng chính < 25 thì kết quả được ghi là thăm dò.
4. Cảnh báo: B_rank lấy từ BM25 trên câu hỏi nên tầng chính có thiên lệch chống lại BM25; đây là lý do phải báo cáo cả ba tầng và chỉ so sánh Graph với RRF/dense trong cùng tầng.
