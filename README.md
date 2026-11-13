# Hybrid Graph-RAG trên ArXiv

Trạng thái ngày 13/11/2026: benchmark v2 assisted 150 câu trên 20.500 paper đã đóng băng; DEV và **test 28 câu đã đánh giá một lần** theo `docs/protocol-test-v2.md`. Kết quả chính: bridge đủ hai gold trong top-10 trên test, Graph MENTIONS 12/14 so với RRF 0/14 (single bằng nhau 14/14). Bridge được dựng bằng đúng loại token mà graph dùng để nối nên kết quả không suy rộng cho multi-hop tổng quát. Các thí nghiệm sau khi mở test đều là **thăm dò trên DEV** (xem mục cuối). **Chưa hoàn tất toàn đề tài/A+.** Xem `docs/tien-do-a-plus.md`.

## Giao diện và kết quả hiện tại

- Demo hỏi đáp và xem DEV: `http://127.0.0.1:8501`. Khởi động bằng `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start_demo.ps1`. Chế độ hỏi đáp chỉ gọi LLM khi bấm nút gửi; hai chế độ còn lại đọc kết quả đã lưu.
- Rà nội dung benchmark: `http://127.0.0.1:8502`, chạy `review_benchmark.py` bằng Streamlit khi cần. Không ghi workbook.
- Chấm người 50 đáp án: `http://127.0.0.1:8503`, chạy `review_answers.py` bằng Streamlit. Ẩn hệ thống/điểm AI, lưu SQLite riêng, không gọi API. Nhãn người của 50 mẫu đã được hai người chấm và đối chiếu (κ trong bản nháp báo cáo); giao diện giữ để xem lại và tái tạo.
- Bản rà trợ lý đã đọc 50 mẫu: `http://127.0.0.1:8503/?view=assistant`, có nhận xét/bằng chứng từng mẫu và đề xuất riêng, không ghi đè đầu ra thí nghiệm hoặc nhãn người. H036 cần chốt tính duy nhất của paper B; H005/H020/H042/H048 là các đáp án lỗi hoặc thiếu hop cần giữ trong kết quả. Các góp ý nhỏ nằm ở bộ lọc riêng.
- Chấm cùng nhóm qua Internet: nhấp đúp `Mo-link-cham-chung.cmd`. Protected Cloudflare Quick Tunnel yêu cầu xác minh email, chỉ dùng các địa chỉ Hill đã chỉ định trong cấu hình local bị Git bỏ qua. Link hiện tại ở `outputs/review-share-link.txt`. Máy Hill cần bật/có mạng; khởi động tunnel mới có thể đổi link. Bản chia sẻ dùng port 8504, ẩn nhận xét trợ lý và tắt agent API; dùng chung SQLite với bản local. Mỗi người nhập tên riêng, nhất quán và bấm Lưu sau mỗi mẫu.
- Báo cáo generation: `results/benchmark_v2_assisted/dev_01/generation_prepared/claude5_controlled/README.md`. Cập nhật offline bằng `python -X utf8 scripts/summarize_generation_v2.py` (trong env_AI).

Trên 31 bridge DEV, RRF đủ hai paper top-10: 2/31; Graph MENTIONS: 26/31. Judge tự động chấm đúng đầy đủ lần lượt 3/31 và 23/31; **chưa hiệu chuẩn người, cấu hình được chọn trên DEV**. Không coi đây là kết quả test hoặc bằng chứng semantic KG USES/PROPOSES. CE reranker và rule typed graph có ablation riêng, gồm cả kết quả âm.

Lượt chính dùng generator claude-sonnet-5 và judge gpt-6-sol, cùng prompt/context cap giữa hai hệ thống. Pilot có trộn nhãn model được giữ riêng và không gộp kết quả chính. Token thực và lịch sử retry nằm trong `token_ledger.json`; chưa có đơn giá nên chưa quy đổi tiền. Không thay `.env`.

Protocol test cố định: `docs/protocol-test-v2.md`. Đã so engine với top-100 đã lưu trên sáu câu DEV, cả năm phương pháp đều khớp. Test 28 câu đã được mở và đánh giá **đúng một lần** (retrieval và generation) sau khi đủ 50 nhãn người; kết quả ở `results/benchmark_v2_assisted/test_01/`. Không chạy lại hoặc chỉnh cấu hình trên test.

Kiểm tra lại offline bằng `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify_experiment_v2.ps1`. Thêm `-CheckEngine` để nạp BGE local và kiểm chứng lại sáu câu DEV. Không gọi API hoặc ghi workbook.

## Tái lập reference bridge v1 (lịch sử)

Tại thư mục gốc repo, chạy PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/reproduce_reference.ps1
```

Lệnh này xây đồ thị từ toàn bộ 20.091 bài, chạy đối chứng và các cấu hình truy xuất trên **DEV**, xuất phân tích lỗi rồi kiểm tra dữ liệu. Không gọi API, không thay đổi Excel, không chạy retrieval trên test. Nó tái sử dụng kết quả BM25/dense/RRF đã có trong `results/bridge_v1/`; không tái huấn luyện hay tạo lại embedding.

Mở giao diện xem bằng chứng:

```powershell
python -m streamlit run app.py --server.address=127.0.0.1
```

Giao diện hiện mặc định v2, có chọn kết quả bridge v1 lịch sử. Chỉ số truy xuất hiển thị là DEV, không phải test. Dùng launch script ở trên để tắt file watcher (tránh quét module vision không dùng khi nạp Transformer).

Kiểm thử:

```powershell
python -m pytest tests -q --basetemp=.pytest_cache/manual-run -o cache_dir=.pytest_cache/manual-cache
```

Trong môi trường hạn chế socket, AppTest của Streamlit cần được chạy ngoài sandbox vì nó tạo kết nối localhost. Không có lệnh kiểm thử nào gọi API thực.

## Dữ liệu và giới hạn diễn giải

- `data/eval/bridge_v1/`: 45 câu bridge, 31 DEV / 14 test. Tách theo thành phần liên thông của cả hai gold paper; không chung gold paper giữa hai phần. Bộ này được lọc theo BM25 B_rank > 10, nên có thiên lệch lựa chọn.
- `data/processed/bridge_corpus_v1.jsonl`: 20.091 paper, phiên bản đóng băng bằng SHA256.
- `data/kg/reference_mentions_v1.json`: đồ thị toàn kho, tạo từ tiêu đề/tóm tắt, không đọc QA hoặc gold. Cạnh **MENTIONS** khác với USES/PROPOSES. Có thể còn tên đồng âm và nhắc đến baseline.
- `data/kg/kg_bridge_dev_v1_v3.graphml`: KG ngữ nghĩa 555 paper, mở rộng theo các paper gold của DEV; chỉ dùng chẩn đoán độ phủ. **Không dùng nó làm bằng chứng so sánh công bằng trên toàn kho.**
- `results/bridge_v1/graph_dev/`: kết quả các cấu hình k=1/2/3, PPR, fusion, đối chứng mở rộng văn bản, cấu hình chọn trên DEV và lỗi cần kiểm tra.
- `results/bridge_v1/reference_artifact_audit.json`: kiểm tra hash Excel, split và vị trí bằng chứng.

## Môi trường

Môi trường chạy: conda env **env_AI** (Windows). Không dùng `.venv-kg`. Kiểm tra bằng `python scripts/check_env.py`. `requirements-experiment.txt` ghi dependency thực nghiệm hiện tại; `requirements-reference.txt` cho reference v1. Chưa cam kết cài sạch trên máy khác. `requirements.txt` cũ là export Anaconda có đường dẫn cục bộ; không dùng nó để cài máy khác.

Dense dùng BGE local snapshot `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`. Embedding cache gốc đã kiểm tra ID/key, kích thước, norm và SHA; revision và cách tạo văn bản của lần tạo cache ban đầu chưa được ghi, nên chưa xác minh độc lập provenance gốc. Không suy ra revision của embedding chỉ từ snapshot hiện có. MiniLM reranker đã có revision/hash trong `data/models/reranker_manifest.json`.

Chấm lại các extraction cache bằng matching một-một (không gọi API):

```powershell
python -X utf8 scripts/rescore_cached_extraction.py
```

## Duyệt benchmark v2

Theo chỉ dẫn “thay cả 3” của Hill, s2006/s2019/q056 đã được thay bằng RAZOR/LTO/Generative Cross-Modal Retrieval. Bộ có 59 single cũ, 46 single mới/thay thế và 45 bridge giữ nguyên. Phiên bản `data/eval/benchmark_v2_assisted/` đã đóng băng **122 DEV / 28 test**, không chung gold paper. Nhãn provenance: 102 rà nguồn bởi trợ lý, 3 quyết định Hill, 45 bridge đã duyệt trước; không công bố thành 150 nhãn người độc lập.

```powershell
python -m streamlit run review_benchmark.py --server.address=127.0.0.1 --server.port=8502
```

Giao diện mặc định lọc **Các câu đã thay**, hiển thị paper và bằng chứng mới cho 3 trường hợp đã chốt. Những bản sửa rõ từ nguồn đã được áp dụng vào bản làm việc, không yêu cầu Hill duyệt lại toàn bộ. Nhật ký người duyệt được giữ riêng. Không thay đổi cấu trúc hoặc nội dung workbook. Xem `data/eval/benchmark_v2_draft/README.md` để biết chi tiết.

Không đưa `.env` vào báo cáo, chat hoặc Git. Các job API cần lưu token, tên model trả về và bảng giá của nhà cung cấp; hiện chưa đủ thông tin để quy đổi chi phí đã chạy sang VND chính xác.

## Thí nghiệm bổ sung 13/11 (thăm dò, chỉ DEV, không mở test)

Ba việc dưới đây đều **thăm dò trên DEV**: test v2 đã mở một lần nên không chạy lại. Kết quả không dùng để chọn cấu hình cuối.

1. **Rerank `bge-reranker-v2-m3` (GPU/Colab)**: `scripts/rerank_bge_export.py` xuất pool 10.094 cặp → chạy `notebooks/colab_rerank_bge.ipynb` (T4, không API) → `scripts/rerank_bge_eval.py --scores .../rerank_bge/scores.jsonl`. Hồi quy trên điểm MiniLM cũ tái tạo đúng từng số.
2. **Thứ tự/độ dài ngữ cảnh** (31 câu bridge, hệ graph): `scripts/prepare_context_order_dev.py` (offline, đã chạy) → `scripts/run_context_order_dev.py` (API: pilot → judge-pilot → dev → judge-dev) → `scripts/summarize_context_order_dev.py`. Dùng đúng generator `claude-sonnet-5` và judge `gpt-6-sol` như baseline.
3. **Môi trường/test**: `python scripts/check_env.py`, rồi `python -m pytest -q`.

### Script thăm dò sau khi test đã mở (exploratory, chỉ DEV hoặc proxy ngoài; không dùng để chọn cấu hình)

`eval_two_hop_baseline.py`, `eval_keyterm_control.py`, `phrase_connectivity_audit.py`, `phrase_expansion_rank.py`, `fetch_citations_s2.py`, `citation_pair_rank.py`, `citation_gating.py`, `citation_cites_leaveout.py`, `eval_cites_fusion.py`, `routing_dev.py`, `rerank_bge_export.py` / `colab_score_bge.py` / `rerank_bge_eval.py`, `prepare_context_order_dev.py` / `run_context_order_dev.py` / `summarize_context_order_dev.py`, và bộ bridge trích dẫn (`select_citation_pairs.py`, `draft_citation_bridge.py`, `make_citation_review_sheet.py`, `finalize_citation_bridge.py`, `eval_citation_bridge.py`; gác lại). Bảng ablation: `make_ablation_table.py`. File engine khóa theo hash trong protocol test không bị sửa bởi các script này.
