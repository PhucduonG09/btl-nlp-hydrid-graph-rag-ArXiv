# Rà 50 đáp án theo nguồn — nhận xét của trợ lý

Đã đọc 50 câu hỏi, đáp án, reference và đối chiếu các nguồn liên quan trong context. Không dùng kiến thức bên ngoài để bù paper bị thiếu. Không thay câu trả lời đã sinh, gold, workbook hoặc nhãn người.

Kết quả rà: {'correct': 33, 'abstained': 13, 'partial': 3, 'incorrect': 1}. Mức ưu tiên: {'none': 40, 'high': 5, 'low': 5}.

## Năm mẫu cần xem trước

| Mẫu | Kết luận | Vấn đề |
|---|---|---|
| H005 | partial | Đúng CvT và các ngưỡng hơn 5 lần/hơn 2,5 lần, nhưng không được chia hai cận dưới để kết luận lợi thế xấp xỉ gấp đôi. Hai thí nghiệm khác dải năng lượng. Abstract cũng không nêu detector noise/calibration/systematics là nguyên nhân. Không có trích dẫn [n] trong đáp án. |
| H020 | incorrect | Lạc đích: câu hỏi cần cách tạo DocFinQA và khó khăn mà tác giả dataset quan sát. Đáp án lại dành phần chính cho constraint verification và adaptive inference của PlanGEN. Các nhận định về PlanGEN có nguồn A nhưng không trả lời B; đây là sai đích/thiếu hop, không tự động là hallucination. |
| H036 | abstained | Từ chối phù hợp vì context không có paper 656 test instances (2310.16534). Quan trọng hơn, câu 'a separate study' không khóa duy nhất paper đánh giá GPT-4V; context đã có một nghiên cứu GPT-4V/Gemini khác (2312.10637). Chưa khẳng định có hai đáp án hoàn toàn tương đương, nhưng mô tả hiện chưa giải thích vì sao chỉ gold B là đúng. Cần Hill chốt tính duy nhất của bridge. |
| H042 | partial | Phần chính đúng 18,2% relative WER, 54% forgetting và retention/plasticity trade-off. Tuy nhiên câu 'all adaptation is evaluated in an on-device, stream-based continual setting' sai phạm vi: abstract có cả full fine-tuning và offline LoRA upper bounds. Không được nói toàn bộ adaptation là on-device. |
| H048 | partial | Đáp án có căn cứ nhưng chỉ mô tả ứng dụng KHRONOS ở paper A. Chưa có construction cụ thể kernel expansions/tensorized modes/superposition và số Poisson của paper giới thiệu B. Phần cảnh báo thiếu originating paper là đúng. Không được lấy 0/10/30% HF availability làm numerical result mà câu hỏi yêu cầu. |

## Năm góp ý nhỏ

- **H015**: Nội dung đủ và giữ đúng tính giả thuyết: SFT đổi tham số nhiều hơn, tác động MLP giữa, freezing chưa kết luận. Tuy nhiên [9] là paper GPQA, không chứng minh kết quả freezing của nghiên cứu RL/SFT; [1] mới là nguồn đúng cho câu đó.
- **H032**: Kết luận chính và caveat hợp nguồn, nhưng 'Temperature matters significantly'/'crucial role' mạnh hơn abstract: nguồn chỉ nói simple parameter adjustments can improve alignment, không báo độ lớn hay kiểm định ý nghĩa thống kê của temperature.
- **H044**: Đúng iterative abstraction, backward-model-irrelevance và Fisher consistency. Câu kết 'maintain their theoretical properties' quá rộng về diễn đạt: kết quả nguồn được nêu cụ thể là Fisher consistency, không mặc nhiên mọi tính chất hay finite-sample guarantee.
- **H047**: Đúng metadataset predictions của năm model trên sáu dataset. Mệnh đề 'rather than retraining the base models themselves' là diễn giải quy trình, không được abstract khẳng định tường minh; nên bỏ để giữ sát nguồn.
- **H050**: Đáp án chính up to 3 lần cho ASR/speech-to-speech translation đúng; các số quarter/2%/4% và codec 1,89 kbps/21,5fps đều có nguồn và có caveat khác task. Tuy nhiên 'degradation growing as the ratio increases' không được các abstract chứng minh như một quan hệ đơn điệu tổng quát.

## Trạng thái cần phân biệt

H036 cần Hill chốt tính duy nhất của câu hỏi gốc; chưa tự sửa phiên bản benchmark đóng băng. Những mẫu khác có lỗi/thiếu hop là kết quả thực nghiệm cần giữ, không phải dòng dữ liệu cần sửa cho tăng điểm.

Có 13 đáp án từ chối: hợp lý với nguồn được cấp ở lượt rà này nhưng không được tính thành 13 câu trả lời đúng. Có 33 đáp án đáp ứng ý chính; trong đó một số có góp ý nhỏ về diễn đạt hoặc trích dẫn, xem từng dòng.

Đây là 50 lượt rà của trợ lý, không thay thế 50 nhãn người độc lập. Chưa điền SQLite human ratings hoặc mở khóa chạy test. Nếu người chấm đã xem gợi ý trợ lý, không mô tả lượt chấm sau đó là chấm mù không tiếp xúc AI.

Xem trên máy: http://127.0.0.1:8503/?view=assistant . Tệp review.json có hash và bằng chứng trích nguyên văn; review.csv là bảng phụ mới, không ghi workbook cũ.
