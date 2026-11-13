"""Kiểm tra nhanh môi trường: in phiên bản các thư viện cần cho thí nghiệm và cảnh báo thư viện thiếu/lệch.
  python scripts/check_env.py
"""
import importlib, platform, sys

REQUIRED = ["numpy", "scipy", "tiktoken", "rank_bm25", "networkx", "openpyxl", "openai", "dotenv", "requests", "pytest"]
HEAVY = ["torch", "sentence_transformers", "transformers", "streamlit"]
OPTIONAL = ["faiss", "neo4j", "sklearn", "pydantic", "yaml", "tqdm"]


def version(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "?")
    except Exception as e:  # noqa: BLE001 - cần bắt mọi lỗi import để báo
        return f"THIẾU/LỖI ({type(e).__name__}: {str(e)[:60]})"


print(f"Python {platform.python_version()} on {platform.system()}")
bad = 0
for title, names in [("Bắt buộc", REQUIRED), ("Nặng (embedding/UI)", HEAVY), ("Tùy chọn", OPTIONAL)]:
    print(f"\n[{title}]")
    for n in names:
        v = version(n)
        flag = "" if not v.startswith("THIẾU") else ("  <-- cần cài" if title != "Tùy chọn" else "  (tùy chọn)")
        bad += bool(flag and title != "Tùy chọn")
        print(f"  {n:22s} {v}{flag}")
try:
    import numpy, scipy
    if int(numpy.__version__.split(".")[0]) >= 2 and tuple(map(int, scipy.__version__.split(".")[:2])) < (1, 14):
        print("\nCẢNH BÁO: scipy < 1.14 với numpy >= 2: nên `pip install -U scipy`.")
except Exception:  # noqa: BLE001
    pass
sys.exit(1 if bad else 0)
