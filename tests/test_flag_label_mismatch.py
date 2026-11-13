import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("flag_label_mismatch", ROOT / "scripts/flag_label_mismatch.py")
flagmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flagmod)


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_flags_only_mismatch_and_is_idempotent_and_keeps_history(tmp_path):
    _write(tmp_path / "generation.jsonl", [
        dict(id="a", system="s", type="bridge", served_model="claude-sonnet-5", model_label_mismatch=False),
        dict(id="b", system="s", type="bridge", served_model="claude-sonnet-5-ta", model_label_mismatch=True),
        dict(id="c", system="s", type="bridge", error="boom", model_label_mismatch=True)])
    _write(tmp_path / "judge.jsonl", [dict(id="b", system="s", quality="correct")])
    assert flagmod.flag(tmp_path) == [("b", "s")]
    gen = [json.loads(l) for l in (tmp_path / "generation.jsonl").read_text().splitlines()]
    assert len(gen) == 4 and gen[1]["served_model"] == "claude-sonnet-5-ta"      # bản ghi cũ còn nguyên
    assert gen[-1]["error"] == "served_label_mismatch_regenerate"
    assert flagmod.flag(tmp_path) == []                                           # chạy lại không đánh dấu thêm
    assert len((tmp_path / "judge.jsonl").read_text().splitlines()) == 2
