# Kết quả generation DEV v2 — tự động, chưa hiệu chuẩn người

122 câu DEV × 2 hệ thống. Generator: claude-sonnet-5; judge: gpt-6-sol. Tập test 28 câu chưa đánh giá.

| Nhóm | Hệ thống | Đúng / tổng | Đúng một phần | Từ chối (judge) | Faithfulness, bỏ đáp án từ chối |
|---|---|---:|---:|---:|---:|
| all | rrf | 62/122 | 43 | 16 | 0.897 |
| all | graph | 93/122 | 24 | 5 | 0.935 |
| single | rrf | 59/91 | 32 | 0 | 0.914 |
| single | graph | 70/91 | 21 | 0 | 0.920 |
| bridge | rrf | 3/31 | 11 | 16 | 0.793 |
| bridge | graph | 23/31 | 3 | 5 | 0.989 |
| legacy_single | rrf | 32/59 | 27 | 0 | 0.891 |
| legacy_single | graph | 44/59 | 15 | 0 | 0.907 |
| new_authored_draft | rrf | 27/32 | 5 | 0 | 0.957 |
| new_authored_draft | graph | 26/32 | 6 | 0 | 0.945 |

Không dùng độ hợp lệ cú pháp trích dẫn làm bằng chứng câu trả lời đúng. Các chỉ số trên do AI chấm, chưa phải kết quả đã xác nhận bởi người.

Chỉ cấp A: {'abstained': 24, 'partial': 7}. Các trường hợp nghi A-only đủ cần rà: [].

Gói chấm người: 50/50 mẫu đã chốt. Mở http://127.0.0.1:8503 để chấm; không gọi API.

Token theo từng nhãn model và mọi lượt thử được lưu trong token_ledger.json; chưa quy đổi tiền. File trả lời gốc, kết quả judge, lịch sử lỗi đều được giữ.

Hạn chế: cấu hình chọn trên DEV; câu bridge có thiên lệch lọc BM25; MENTIONS chưa phải quan hệ ngữ nghĩa USES/PROPOSES; nguồn nhãn assisted không thay thế annotation người.
