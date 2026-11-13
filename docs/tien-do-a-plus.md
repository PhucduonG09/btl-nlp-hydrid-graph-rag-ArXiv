# Tiến độ đối chiếu tiêu chí A+ — 13/11/2026

Nguồn định hướng: `D:\Study\NLP\Tổng quan & tiêu chí A+.pdf`, tài liệu Hill cung cấp, chưa xác nhận là rubric chính thức. Bản trạng thái 09/11 được giữ ở `tien-do-a-plus-09-11.md`; lịch sử trước generation ở `tien-do-a-plus-before-generation.md`. Bản nháp báo cáo đầy đủ (số liệu, hạn chế, checklist) nằm trong Project "NLP - Natural Language Processing": `claude/bao-cao-cuoi-ky-ban-nhap.md`.

## Đã hoàn thành và kiểm chứng

- [x] Benchmark **v2 assisted**: 150 câu = 105 single + 45 bridge, corpus 20.500 paper; 122 DEV / 28 test không chung gold paper; manifest khóa hash. Không phải annotation người độc lập (102 câu rà nguồn bởi trợ lý, 3 câu thành viên quyết định, 45 bridge duyệt trước).
- [x] Baseline BM25/BGE/RRF, mở rộng văn bản, graph k=1/2/3/PPR/fusion: 43 cấu hình DEV; cấu hình khóa ở `dev_01/selected_config.json`. Bảng ablation: `results/analysis/ablation_tables.md` (tạo bằng `scripts/make_ablation_table.py`).
- [x] Đồ thị MENTIONS toàn corpus (33.643 thực thể / 58.026 cạnh), không đọc QA/gold; không phải KG USES/PROPOSES.
- [x] Chấm người 50 mẫu bởi hai người, nhãn cuối sau đối chiếu nguồn; hiệu chuẩn judge (κ `quality` 0,58, `support` 0,70).
- [x] **Test v2 (28 câu) đã mở đúng một lần**: retrieval + generation theo protocol `docs/protocol-test-v2.md`. TEST bridge đủ hai gold trong top-10: Graph 12/14 so với RRF 0/14.
- [x] Reranker: MiniLM (DEV) và `bge-reranker-v2-m3` (DEV, Colab T4, chưa ghim revision): cả hai đều làm giảm bridge All@10; không dùng trong cấu hình cuối.
- [x] Thứ tự/độ dài ngữ cảnh (DEV, 30 câu bridge): đảo/sandwich không đổi số câu đúng; cắt top-5 giảm 50% ngữ cảnh nhưng mất 6 câu đúng ròng (p=0,070).
- [x] Kiểm chứng độc lập (thăm dò sau khi test mở): two-hop không graph, từ hiếm, cụm khái niệm, trích dẫn thật (Semantic Scholar), CITES, routing, gating. Kết luận: lợi thế của Graph đặc thù cho liên kết qua tên riêng/viết tắt chung.
- [x] Hướng KG LLM quy mô lớn: **quyết định dừng** (không có bằng chứng độc lập ủng hộ); không còn là việc tiếp theo.
- [x] Nhật ký token, demo hỏi đáp có trích dẫn, nhiều bộ kiểm thử. Trong sandbox Python 3.10: 78 test đạt, bỏ qua 3 file giao diện cần streamlit.

## Kết quả chính (TEST, 28 câu; cấu hình khóa từ DEV)

| Chỉ số | RRF | Graph MENTIONS |
|---|---:|---:|
| Single đủ gold trong top-10 | 14/14 | 14/14 |
| Bridge đủ hai gold trong top-10 | 0/14 | 12/14 |
| Bridge trả lời thực chất (không từ chối) | 2/14 | 12/14 |

Chênh lệch ở bridge do truy xuất (gold có nằm trong ngữ cảnh hay không), không do bước sinh; ở single hai hệ thống như nhau. **Bridge được dựng bằng đúng loại token mà graph dùng để nối, rồi lọc theo BM25**, nên kết quả chỉ chứng minh graph giải được bridge một bước qua tên riêng; xem mục 9 của bản nháp báo cáo.

## Chưa hoàn tất / cần người làm

- [ ] Viết báo cáo cuối và slide từ bản nháp; điền tên nhóm/thành viên/giảng viên/ngày; kiểm tra tên-năm-tác giả mọi paper trích dẫn; dẫn chứng học thuật về hallucination.
- [ ] Chạy `python -m pytest -q` trong **env_AI** (đủ 3 file giao diện) và `python scripts/check_env.py`; ghim phiên bản vào `requirements-experiment.txt` bằng `pip freeze`.
- [ ] Ghim revision của `bge-reranker-v2-m3` nếu muốn tái lập chặt (hiện `resolved_revision` null).
- [ ] Rà 20 witness quote của judge (14 không nguyên văn, 4 ghép bằng dấu ba chấm) nếu đưa phân tích faithfulness vào báo cáo.
- [ ] Thống nhất cách ghi độ lệch so với kế hoạch FAISS + Neo4j (hiện numpy + đồ thị in-memory); nếu giảng viên yêu cầu, thêm demo xuất Neo4j (`src/kg/neo4j_io.py`).
- [ ] Xoay API key đã lộ; xóa `_to_delete/`; không đưa `.env` vào Git hay báo cáo.
- [ ] Chưa trả lời được: "graph có giúp multi-hop tổng quát hay không". Cần bộ bridge qua liên kết tự nhiên (trích dẫn) quy mô đủ lớn có duyệt người; hướng này đã thử ở quy mô nhỏ và gác lại (`claude/ke-hoach-bo-bridge-trich-dan-2026-11-12.md` trong Project).

**Chưa tick A+.** Các mục trên cần người hoặc cần quyết định của nhóm.
