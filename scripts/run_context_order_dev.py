"""Chạy sinh/judge cho thí nghiệm thứ tự ngữ cảnh bằng CHÍNH runner của baseline (run_generation_v2.py), chỉ đổi thư mục job.

Thứ tự bắt buộc (mỗi lệnh gọi API, bạn chạy):
  python scripts/run_context_order_dev.py --stage pilot       --generator-model <model>
  python scripts/run_context_order_dev.py --stage judge-pilot
  python scripts/run_context_order_dev.py --stage dev         --generator-model <model>
  python scripts/run_context_order_dev.py --stage judge-dev
Giống baseline: cần LLM_MODEL (hoặc --generator-model) và LLM_JUDGE_MODEL khác họ model. Chạy lại được (có resume).
"""
import importlib.util, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("run_generation_v2", ROOT / "scripts/run_generation_v2.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.BASE = Path(os.environ.get("CTX_ORDER_DIR", ROOT / "results/benchmark_v2_assisted/dev_01/generation_prepared/context_order"))
if __name__ == "__main__":
    runner.main()
