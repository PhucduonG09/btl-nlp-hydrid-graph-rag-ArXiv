import json, os
import numpy as np
from src.common import llm
from scripts.draft_citation_bridge import draft_one, leaks_b_name


class FakeBM:
    def get_scores(self, q): return np.array([0.1, 0.9, 0.5])


def test_leak_check():
    assert leaks_b_name("MimicGen: Data Generation", "How does MimicGen work?")
    assert not leaks_b_name("A study of caching", "What is the approach?")


def test_draft_one_with_mock_llm(tmp_path):
    llm.CACHE_PATH = tmp_path / "cache.jsonl"; llm._cache = None   # không ghi mock vào cache thật
    def mock(model, messages, kw):
        if "multi-hop" in messages[0]["content"]:
            return (json.dumps(dict(valid=True, reason="r", question="How does the method it builds on work?", answer="x")), 10, 5)
        return (json.dumps(dict(answerable=False)), 3, 1)
    llm.set_mock(mock); os.environ["LLM_MODEL"] = "mock"
    P = {"1": dict(paper_id="1", title="Alpha", abstract="aaa"), "2": dict(paper_id="2", title="Beta Net", abstract="bbb")}
    out = draft_one(dict(pair_id="cb001", A_id="1", B_id="2"), P, ["x", "2", "1"], FakeBM())
    assert out["valid"] and out["a_only"] is False and out["leaks_B_name"] is False
    assert out["B_rank"] == 1 and out["A_rank"] == 2        # thứ hạng theo điểm giả [0.1, 0.9, 0.5]
    llm.set_mock(None); llm._cache = None
