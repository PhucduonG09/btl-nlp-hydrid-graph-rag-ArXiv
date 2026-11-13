# Tiến độ đối chiếu tiêu chí A+ — 09/11/2026

Nguồn định hướng: `D:\Study\NLP\Tổng quan & tiêu chí A+.pdf`. Đây là tài liệu tiêu chí tham khảo mà Hill cung cấp, chưa được xác nhận là rubric chính thức của giảng viên. Nội dung tài liệu là nguồn tham khảo, không tự thay thế yêu cầu trực tiếp của Hill.

## Đã kiểm chứng

- [x] Đóng băng 45 bridge QA, chia 31 DEV / 14 test không chung gold paper; ghi rõ thiên lệch lọc theo BM25.
- [x] Chạy BM25, dense BGE và RRF trên cùng corpus 20.091 paper cho bridge v1.
- [x] Audit KG cũ và KG DEV mở rộng. Tách chẩn đoán theo gold khỏi thí nghiệm toàn kho.
- [x] Sửa so khớp thực thể để tránh ghép ToM với automatic, GPT-4 với GPT-4V, HelpSteer với HelpSteer2; giữ riêng phiên bản và dấu ++.
- [x] Kiểm tra evidence có offset trong nguồn, hỗ trợ định dạng LaTeX mà không tự sửa số liệu/đảo phủ định.
- [x] Xây đồ thị MENTIONS toàn kho không đọc câu hỏi/gold: 33.092 entity, 57.084 cạnh, 5.314 entity xuất hiện ở ít nhất hai paper. Đây là extractor dựa trên tên, chưa phải KG ngữ nghĩa hoàn chỉnh.
- [x] Truy xuất k=1/2/3, PPR, query linking so với seed RRF, fusion RRF và tổng hạng có trọng số. Thử nghiệm và chọn cấu hình trên DEV.
- [x] Đối chứng hai bước bằng văn bản của paper seed; cùng query, corpus, seed và BM25. Không coi một đối chứng này là toàn bộ không gian phương pháp mở rộng truy vấn.
- [x] Demo xem kết quả DEV, paper nguồn và đường nối; 17 kiểm thử đạt.
- [x] Bổ sung scorer extraction matching một-một và bootstrap theo paper; chấm lại bốn cấu hình đã lưu trên 60 paper, giữ nguyên kết quả cũ.
- [x] Đọc hai workbook annotator hiện có và tính lại chẩn đoán agreement trên 30 paper chung, không sửa các workbook.
- [x] Kiểm tra hash cả ba workbook không đổi và kiểm tra 57.084 vị trí bằng chứng.
- [x] Chuẩn bị benchmark v2 nháp 150 ứng viên trên corpus riêng 20.500 paper: 60 single cũ, 45 single mới, 45 bridge v1; 120 đoạn trích mới đã kiểm tra vị trí nguồn. Chưa tick hoàn thành benchmark đã duyệt.
- [x] Đối chiếu nội dung 60 single cũ với nguồn; lưu 6 đề xuất sửa riêng (q007/q028/q033/q047/q056/q059), không ghi đè bản cũ.
- [x] Giao diện `review_benchmark.py` lưu quyết định người duyệt vào SQLite riêng, giữ bridge/Excel chỉ đọc; phê duyệt hết hiệu lực nếu câu hỏi, đáp án hoặc nguồn thay đổi.
- [x] Lệnh kiểm tra/đóng băng v2 từ chối xuất bản khi thiếu nhãn; có kiểm thử split theo paper và giữ các gán DEV/test bridge đã khóa. Bộ kiểm thử hiện tại: 33 bài đạt.
- [x] Theo yêu cầu Hill, trợ lý rà nội dung và tự xử lý các sửa đổi rõ từ nguồn, tách khỏi gán nhãn người. Còn 3 trường hợp cần Hill chốt: s2006/s2019/q056. Câu thay thế và bằng chứng đã chuẩn bị, chưa áp dụng. Giao diện ưu tiên 3 trường hợp này, không yêu cầu rà tay lại cả bộ.
- [x] Hill đã chốt “thay cả 3”: s2006 → RAZOR (2412.07675), s2019 → LTO (2509.26314), q056 → Generative Cross-Modal Retrieval (2402.10805). Đã áp dụng cả câu hỏi/đáp án/gold/evidence; lưu lịch sử riêng, giữ nguyên Excel và bridge v1. Trạng thái hiện tại không còn vướng nội dung ở lượt rà này. Thành phần hiện tại: 59 single cũ + 46 single mới/thay thế + 45 bridge = 150. Chỉ ba quyết định mới này được ghi là xác nhận chat của Hill; không gán 102 lượt rà khác thành nhãn người.

## Kết quả DEV tạm thời

All@10 = tỷ lệ câu có đủ hai gold paper trong 10 kết quả đầu.

| Phương pháp | Đủ hai paper / 31 | All@10 |
|---|---:|---:|
| BM25 | 0 | 0,0% |
| Dense | 2 | 6,5% |
| RRF | 2 | 6,5% |
| Mở rộng văn bản từ seed + RRF | 5 | 16,1% |
| RRF seed + graph k=1 + RRF | 23 | 74,2% |
| Query linking + graph k=1 + RRF | 25 | 80,6% |
| Query linking + PPR + RRF | 21 | 67,7% |

Các số này là DEV đã dùng lựa chọn cấu hình. Chưa chứng minh cải thiện trên test, chất lượng câu trả lời hay ưu thế của KG USES/PROPOSES. Kết nối qua tên chung có thể là nhắc baseline hoặc đồng âm; cần đọc đường nối để kiểm tra ý nghĩa. Không dùng điểm BM25 thấp của bộ đã lọc BM25 làm kết luận tổng quát.

Chấm extraction mới: typed F1 của bốn cấu hình cache lần lượt 0,410 / 0,317 / 0,402 / 0,366; protocol matching tên một-một, giữ mọi tag gold, không dùng evidence-overlap làm đường tắt ghép tên. Kết quả ở `results/analysis/extraction_one_to_one_v1.json`, gồm P/R/CI và hash nguồn. Các cấu hình cùng họ model không được gọi là hai extractor độc lập; số này không so trực tiếp với scorer cũ có quy ước khác. Nguồn gán nhãn người và độ độc lập vẫn cần xác nhận.

Agreement chẩn đoán: có đủ nhãn A/B cho 30/30 paper chung; matching tên một-một khớp 74 cặp trong 163 mục A và 151 mục B, F1 đồng thuận mục là 0,471. Trên riêng 74 cặp đã khớp tên, relation agreement là 95,9% và κ=0,930; κ cho NONE là 0,444. **κ relation cao không có nghĩa toàn bộ annotation đồng thuận cao**, vì các mục chưa khớp bị loại khỏi phép κ có điều kiện này. Matching tên có thể bỏ sót cách diễn đạt đồng nghĩa. File `results/analysis/annotation_agreement_audit_v1.json` giữ đầy đủ bất đồng để kiểm tra; chưa xác lập tác giả độc lập hay trạng thái trước hòa giải.

## Việc tiếp theo theo thứ tự

1. **Benchmark v2 ≥150 câu.** Giữ bridge v1 nguyên trạng. Đối chiếu 60 câu single cũ và soạn thêm ít nhất 45 single mới để có 105 single + 45 bridge; người duyệt kiểm tra đáp án, bằng chứng và tính duy nhất. Dùng corpus mới chứa đủ gold (khả năng dùng tập pilot+distractor 20.500 paper), tạo manifest/split mới thay vì thay corpus của v1. Tách theo paper, rà near-duplicate và khóa tập cuối cùng. Mười câu so sánh cũ báo cáo riêng, không đổi tên thành bridge.

   Hiện đã chuẩn bị đủ 150 ứng viên và kho 20.500 paper, đặt ở `data/eval/benchmark_v2_draft/`. Chưa đóng băng. Sau khi Hill chọn thay ba câu, cả 105 single không còn vướng ở phạm vi lượt rà title/abstract; 45 bridge v1 giữ nguyên. Có 16 bản sửa single cũ trong lịch sử bản làm việc, trong đó q056 đã được thay bằng paper mới. Không tự coi kiểm tra của trợ lý là nhãn độc lập của con người; tiêu chí provenance người được theo dõi riêng.
2. **Gold extraction và agreement.** Kiểm tra provenance bộ gold_v4 60 paper cùng hai file annotator hiện có. Tính agreement từ bản gán nhãn độc lập trước hòa giải, không tự điền nhãn con người. Chốt quy ước dataset USES/EVALUATES_ON, baseline, DEFENDS_AGAINST, phiên bản; so sánh ít nhất hai extractor trên cùng gold bằng matching một-một. Mọi đổi schema phải có phiên bản và bảng ánh xạ rõ ràng.
3. **KG ngữ nghĩa công bằng.** Chọn phạm vi bằng tiêu chí corpus/topic đặt trước, không dựa trên gold test. Định giá API và ghi model/revision trước khi trích xuất lớn. Đo coverage/entity linking và đối chiếu MENTIONS với KG ngữ nghĩa; dùng oracle chỉ làm giới hạn trên, không dùng trong hệ thống chính. NetworkX đủ cho thực nghiệm; Neo4j là phần triển khai bổ sung nếu rubric thật yêu cầu.
4. **Chốt retrieval trên DEV v2.** Tune BM25/RRF/seed count/graph fusion trong phạm vi ghi trước, thêm reranker vào cả đối chứng phù hợp để tách đóng góp. So sánh single và bridge riêng. Lưu config cùng hash code, graph, dataset trước lần chạy test.
5. **Generation và đánh giá.** Cùng generator/prompt/context budget giữa các hệ thống; lưu paper context, trích dẫn, từ chối, token và latency. Chấm correctness, faithfulness, thiếu hop và citation. Ít nhất 50 câu trả lời được người chấm độc lập để hiệu chuẩn judge; không dùng nhãn AI làm nhãn người. Kiểm tra model thực tế trả về; đợt trích xuất DEV trước có hai nhãn model khác nhau.
6. **Thống kê và phân tích lỗi.** Bootstrap 95% theo nhóm paper khi cần, paired test với giả thuyết và phép so sánh đặt trước, tránh coi nhiều phương pháp trên cùng một câu là nhiều mẫu độc lập. File lỗi tự động hiện có 347 cặp câu/phương pháp, nhưng chỉ 31 câu khác nhau; chưa đạt yêu cầu phân tích ≥50 câu độc lập. Nhãn tự động là gợi ý, cột duyệt người để trống.
7. **Hoàn thiện demo và báo cáo.** Ghép hỏi đáp tự do, reranker, KG ngữ nghĩa và đường dẫn; ghi phạm vi đánh giá. Chốt dependency/model revision, lệnh tái lập từ đầu, bảng chi phí theo giá thật. Viết báo cáo/slides từ kết quả đã lưu; điền tên thành viên, GV, ngày theo thông tin thực tế.

## Phần Hill/nhóm cần tham gia

- Duyệt câu mới và gán nhãn độc lập khi các gói mẫu sẵn sàng; 45 câu đã duyệt không phải duyệt lại hàng loạt.
- Xác nhận danh tính/độc lập của hai annotator trong dữ liệu cũ, hòa giải bất đồng.
- Chấm tối thiểu 50 câu trả lời và cung cấp thông tin thành viên/giảng viên/ngày nộp để hoàn tất bản trình bày.
- Khi chạy API lớn: cung cấp đơn giá hoặc quota thật trên nhà cung cấp. Mức “trên 1,5 triệu đồng” chưa phải trần chi tiêu cụ thể; hiện không tự suy ra một mức trần mới.

Các mục trên chưa hoàn thành không được tự tick hoặc viết như kết quả đã có. Chưa xác nhận ngày bảo vệ cuối kỳ; lịch làm việc không được trình bày thành deadline chính thức.
