"""Kiểm thử pipeline rerank GPU: export -> chấm điểm (scorer giả) -> eval. Không cần model, GPU hay mạng."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


export = _load("rerank_bge_export")
colab = _load("colab_score_bge")
ev = _load("rerank_bge_eval")


def test_pool_is_ordered_union_without_duplicates():
    row = dict(rankings=dict(r=list("abcd"), g=list("cxya")))
    assert export.pool_for(row, dict(rrf="r", graph="g"), depth=3) == list("abcxy")


def test_input_key_depends_on_text_and_question():
    k = export.input_key("q", ["a"], ["t"])
    assert k != export.input_key("q", ["a"], ["t2"]) and k != export.input_key("q2", ["a"], ["t"])


def _rows():
    return [dict(id="q1", question="q", key="k1", pool=[dict(id="a", text="aa"), dict(id="b", text="bbbb")]),
            dict(id="q2", question="q", key="k2", pool=[dict(id="c", text="c")])]


def test_score_rows_resumes_and_checks_lengths():
    calls = []

    def predict(q, texts):
        calls.append(texts)
        return [float(len(t)) for t in texts]
    first = colab.score_rows(_rows(), predict)
    assert [r["scores"] for r in first] == [dict(a=2.0, b=4.0), dict(c=1.0)]
    again = colab.score_rows(_rows(), predict, done_keys={"k1"})  # chạy tiếp: bỏ qua câu đã xong
    assert [r["id"] for r in again] == ["q2"]


def test_score_rows_rejects_bad_scores():
    for bad in ([1.0], [float("nan"), 1.0]):
        try:
            colab.score_rows(_rows()[:1], lambda q, t, bad=bad: bad)
        except AssertionError:
            continue
        raise AssertionError("phải từ chối điểm sai độ dài hoặc NaN")


def test_variants_keep_pool_and_rerank_changes_order():
    originals = dict(rrf=list("abcd"), graph=list("dcba"))
    scores = dict(a=0.0, b=1.0, c=2.0, d=3.0)
    v = ev.build_variants(originals, scores)
    assert v["rrf_base"] == list("abcd") and v["rrf_ce"] == list("dcba")
    assert all(sorted(r) == list("abcd") for r in v.values())  # không thêm/bớt tài liệu


def test_mcnemar_exact_known_values():
    assert ev.mcnemar_exact(0, 0) == 1.0
    assert abs(ev.mcnemar_exact(0, 5) - 0.0625) < 1e-12  # 2 * (1/32)
    assert ev.mcnemar_exact(3, 3) == 1.0


def test_success_definitions():
    bridge = dict(type="bridge", hop1_id="a", hop2_id="b", gold_ids=["a", "b"])
    assert ev.success(bridge, list("xab")) and not ev.success(bridge, list("xa") + ["z"] * 9 + ["b"])
    single = dict(type="single", gold_ids=["a"])
    assert ev.success(single, list("ab")) and not ev.success(single, list("ba"))
