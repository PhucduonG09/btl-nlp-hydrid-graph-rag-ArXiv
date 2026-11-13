from src.data.citations import parse_batch


def test_parse_batch_keeps_only_in_corpus_and_drops_self():
    ids = ["1.1", "2.2", "3.3"]
    data = [dict(references=[dict(externalIds=dict(ArXiv="2.2")), dict(externalIds=dict(ArXiv="9.9")), dict(externalIds=None), None, dict(externalIds=dict(ArXiv="1.1"))]),
            None, dict(references=None)]
    edges, found = parse_batch(ids, data, {"1.1", "2.2", "3.3"})
    assert edges == [("1.1", "2.2")] and found == {"1.1", "3.3"}
