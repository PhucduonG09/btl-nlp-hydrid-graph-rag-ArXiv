import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.ir_metrics import recall_at_k, mrr_at_k, ndcg_at_k, hit_all_at_k
from src.kg.aliases import match, canon, is_acronym, merge_acronyms
from src.kg import llm_extract
from src.common import llm


def test_metrics():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 2) == 0.5
    assert mrr_at_k(["x", "b"], ["b"]) == 0.5
    assert abs(ndcg_at_k(["b", "a"], ["b"]) - 1.0) < 1e-9
    assert hit_all_at_k(["a", "b"], ["a", "b"], 2) == 1.0 and hit_all_at_k(["a"], ["a", "b"], 5) == 0.0


def test_alias_match():
    assert match("NER", "named entity recognition", loose=True)
    assert not match("NER", "named entity recognition", loose=False)
    assert match("DTW", "Dynamic Time Warping", loose=True)
    assert match("LLMs", "large language models", loose=True)
    assert match("The BERT model", "bert model")
    assert canon("Large Language Models (LLMs)") == "large language models"
    assert not match("cvt", "mlp", loose=True)
    assert is_acronym("ner", "named entity recognition") and not is_acronym("ner", "neural network")
    m = merge_acronyms({"ner": 2, "named entity recognition": 5})
    assert m["ner"] == m["named entity recognition"] == "named entity recognition"


def test_extract_tolerant(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "c.jsonl"); llm._cache = None
    resp = {"relations": [
        {"relation": "proposes", "target": "AdaFish", "target_type": "method", "evidence": "we introduce AdaFish"},
        {"relation": "USES", "target": "X", "target_type": "weird", "evidence": "not in text"},
        {"relation": "BOGUS", "target": "Y", "target_type": "Method", "evidence": "we introduce AdaFish"},
        {"relation": "USES", "target": "UMAP", "target_type": "resource", "evidence": "using   UMAP"}]}
    llm.set_mock(lambda m, msgs, kw: ("```json\n" + json.dumps(resp) + "\n```", 10, 5))
    p = {"paper_id": "1", "title": "T", "abstract": "Here we introduce AdaFish, using UMAP."}
    o = llm_extract.extract_one(p, "v3", model="m")
    assert [(r["relation"], r["target"], r["target_type"]) for r in o["relations"]] == \
        [("PROPOSES", "AdaFish", "Method"), ("USES", "UMAP", "Dataset")]
    assert o["dropped_no_evidence"] == 1 and o["n_invalid_items"] == 1
    llm.set_mock(lambda m, msgs, kw: ("not json", 1, 1))
    assert llm_extract.extract_one({**p, "paper_id": "2", "abstract": "zzz"}, "v3", model="m")["error"] == "invalid_json"
    llm.set_mock(None)


def test_cache_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "c.jsonl"); llm._cache = None
    calls = []
    llm.set_mock(lambda m, msgs, kw: (calls.append(1) or '{"relations": []}', 3, 2))
    papers = [{"paper_id": str(i), "title": "t", "abstract": "a"} for i in range(5)]
    out = tmp_path / "o.jsonl"
    llm_extract.run_extraction(papers, out, "v3", model="m", workers=3)
    n1 = len(calls)
    llm_extract.run_extraction(papers, out, "v3", model="m", workers=3)
    assert len(calls) == n1 and len(out.read_text().splitlines()) == 5
    llm.set_mock(None)


def test_flaky_401_is_retried(tmp_path, monkeypatch):
    import openai
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "c.jsonl"); llm._cache = None
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    llm.set_mock(None); llm._abort.clear(); llm._consec_fatal = 0
    calls = {"n": 0}

    class FakeCompletions:
        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] <= 2:
                e = openai.APIStatusError.__new__(openai.APIStatusError)   # tránh phụ thuộc cách dựng response
                Exception.__init__(e, "Invalid token"); e.status_code = 401
                raise e
            msg = type("M", (), {"content": "ok"})()
            return type("R", (), {"choices": [type("C", (), {"message": msg})()], "usage": None, "model": "m"})()
    fake = type("F", (), {"chat": type("Ch", (), {"completions": FakeCompletions()})()})()
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    text, usage = llm.chat([{"role": "user", "content": "hi"}], model="m")
    assert text == "ok" and calls["n"] == 3 and not llm._abort.is_set()


def test_retry_without_temperature(monkeypatch):
    import openai
    from types import SimpleNamespace as NS
    err = openai.BadRequestError.__new__(openai.BadRequestError)   # tránh phụ thuộc phiên bản httpx
    Exception.__init__(err, "model does not support non-default temperature")
    err.status_code = 400
    seen = []

    def create(**kw):
        seen.append("temperature" in kw)
        if "temperature" in kw:
            raise err
        return NS(choices=[NS(message=NS(content="ok"))], usage=NS(prompt_tokens=3, completion_tokens=1), model="served")
    fake = NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr(llm, "_get_client", lambda: fake)
    llm._NO_TEMP.clear()
    text, usage = llm._call_with_retry("claude-x", [{"role": "user", "content": "hi"}], {"temperature": 0.0})
    assert text == "ok" and seen == [True, False] and "claude-x" in llm._NO_TEMP


def test_extraction_output_cap_has_separate_cache_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "caps.jsonl"); llm._cache = None
    calls = []
    llm.set_mock(lambda model, messages, kw: (calls.append(kw["max_tokens"]) or '{"relations": []}', 3, 2))
    paper = {"paper_id": "cap-test", "title": "T", "abstract": "A"}
    try:
        llm_extract.extract_one(paper, model="m", max_tokens=1800)
        llm_extract.extract_one(paper, model="m", max_tokens=900)
        cached = llm_extract.extract_one(paper, model="m", max_tokens=900)
        assert calls == [1800, 900] and cached["usage"]["cached"]
    finally:
        llm.set_mock(None)
