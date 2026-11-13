"""Đường ống thí nghiệm thứ tự ngữ cảnh với LLM giả, trong thư mục tạm: không API, không ghi vào cache/kết quả thật."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "results/benchmark_v2_assisted/dev_01/generation_prepared/context_order"
pytestmark = pytest.mark.skipif(not (PREPARED / "plan.json").exists(), reason="chưa chạy prepare_context_order_dev.py")


def _mock(model, messages, kw):
    if messages[0]["content"].startswith("Evaluate an answer"):
        return json.dumps(dict(quality="correct", relevance=5, claims=[])), 10, 5
    return "Mock answer [1].", 10, 5


def _run(monkeypatch, tmp, argv):
    from src.common import llm
    monkeypatch.setenv("CTX_ORDER_DIR", str(tmp))
    monkeypatch.setenv("LLM_MODEL", "claude-mock")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "gpt-mock")           # judge phải khác họ model với generator
    monkeypatch.setattr(llm, "CACHE_PATH", tmp / "unused_cache.jsonl")
    monkeypatch.setattr(llm, "_cache", None)
    llm.set_mock(_mock)
    monkeypatch.setattr(sys, "argv", ["run_context_order_dev.py"] + argv)
    spec = importlib.util.spec_from_file_location("run_ctx_wrapper", ROOT / "scripts/run_context_order_dev.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.runner.main()
    finally:
        llm.set_mock(None)


def test_full_mock_pipeline_and_summary(monkeypatch, tmp_path, capsys):
    for f in ["plan.json", "jobs_dev.jsonl", "pilot_jobs.jsonl"]:
        shutil.copy(PREPARED / f, tmp_path / f)
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["n_jobs"] == 93 and plan["paid_calls"] == 0 and plan["checks"]["truncated_sources"] == 0
    real_cache_before = (ROOT / "data/cache/llm.jsonl").stat().st_mtime if (ROOT / "data/cache/llm.jsonl").exists() else None
    _run(monkeypatch, tmp_path, ["--stage", "pilot"])
    _run(monkeypatch, tmp_path, ["--stage", "judge-pilot"])
    _run(monkeypatch, tmp_path, ["--stage", "dev"])
    _run(monkeypatch, tmp_path, ["--stage", "judge-dev"])
    gen = [json.loads(l) for l in (tmp_path / "generation.jsonl").read_text(encoding="utf-8").splitlines()]
    jud = [json.loads(l) for l in (tmp_path / "judge.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({(r["id"], r["system"]) for r in gen}) == 93 and not any(r.get("error") for r in gen + jud)
    assert {r["system"] for r in gen} == {"graph_reverse", "graph_sandwich", "graph_top5"}
    # dev tái dùng kết quả pilot, không gọi lại: tổng số bản ghi = số job duy nhất + job pilot lặp 0 lần
    assert len(gen) == 93
    # tổng hợp
    monkeypatch.setenv("CTX_ORDER_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["summarize"])
    spec = importlib.util.spec_from_file_location("sum_ctx", ROOT / "scripts/summarize_context_order_dev.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    report = json.loads((tmp_path / "context_order_summary.json").read_text(encoding="utf-8"))
    assert report["n_bridge_used"] == 31 and report["excluded"] == {} and set(report["paired_vs_rank"]) == {"graph_reverse", "graph_sandwich", "graph_top5"}
    assert report["test_evaluated"] is False
    after = (ROOT / "data/cache/llm.jsonl").stat().st_mtime if (ROOT / "data/cache/llm.jsonl").exists() else None
    assert real_cache_before == after, "test không được ghi vào cache LLM thật"


def test_summary_excludes_question_with_invalid_judge(monkeypatch, tmp_path):
    for f in ["plan.json", "jobs_dev.jsonl", "pilot_jobs.jsonl"]:
        shutil.copy(PREPARED / f, tmp_path / f)
    for stage in ["pilot", "judge-pilot", "dev", "judge-dev"]:
        _run(monkeypatch, tmp_path, ["--stage", stage])
    with (tmp_path / "judge.jsonl").open("a", encoding="utf-8") as f:       # judge suy biến ở 1 biến thể của bn009
        f.write(json.dumps(dict(id="bn009", system="graph_reverse", type="bridge", error="invalid_judge_json_or_schema")) + "\n")
    monkeypatch.setenv("CTX_ORDER_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["summarize"])
    spec = importlib.util.spec_from_file_location("sum_ctx2", ROOT / "scripts/summarize_context_order_dev.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    report = json.loads((tmp_path / "context_order_summary.json").read_text(encoding="utf-8"))
    assert report["n_bridge_used"] == 30 and list(report["excluded"]) == ["bn009"]   # loại khỏi mọi hệ thống để ghép cặp cùng tập câu
