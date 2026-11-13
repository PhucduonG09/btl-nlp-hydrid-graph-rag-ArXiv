# Chạy phần Giữa kỳ (chạy từ thư mục gốc repo, trong môi trường Python của bạn)

0. Cài thư viện (một lần): `pip install -r requirements-gk.txt`
   Mở file `.env`, dán key vào sau `LLM_API_KEY=` (không dán key vào notebook hay chat).
   Kiểm tra nhanh: `python -m pytest -q tests/test_gk.py` (không gọi API).

1. Chấm lại extractor trên 20 paper dev (v2 là kết quả bạn đã có):
   python scripts/eval_extraction.py --pred data/kg/dev_extract_out_v2.jsonl --label v2
   python scripts/kg_extract.py --input data/eval/dev_extract.jsonl --out data/kg/dev_extract_out_v3.jsonl --prompt v3
   python scripts/eval_extraction.py --pred data/kg/dev_extract_out_v3.jsonl --label v3
   python scripts/eval_extraction.py --pred data/kg/dev_extract_out_v3.jsonl --label v3+bo-chung --drop-generic

2. Trích xuất 500 paper (resume được; chạy lại cùng lệnh nếu bị ngắt):
   python scripts/kg_extract.py --input data/processed/pilot_papers.jsonl --out data/kg/pilot_extract_v3.jsonl --prompt v3 --workers 6
   Nên thử --limit 20 trước để xem chi phí thật.

3. Xây KG và thống kê:
   python scripts/build_kg.py --extract data/kg/pilot_extract_v3.jsonl --tag v3

4. Bộ câu hỏi:
   python scripts/make_qa.py generate --triples data/kg/triples_v3.jsonl --n-single 60 --n-multi 40
   (mở data/eval/qa_review.csv bằng Excel: keep=0 để loại câu dở, sửa cột question nếu cần; rà ít nhất 30 câu)
   python scripts/make_qa.py finalize

5. Baseline truy xuất (BM25, Dense, RRF):
   python scripts/run_baselines.py

6. Sinh câu trả lời và chấm (khoảng 50 câu x 3 hệ thống):
   python scripts/eval_generation.py --n 50 --k 5

Kết quả nằm trong results/ và data/kg/. Mọi lời gọi LLM được cache ở data/cache/llm.jsonl (chạy lại không tốn tiền).
