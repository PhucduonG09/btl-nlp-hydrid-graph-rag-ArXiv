from scripts.select_citation_pairs import select


def test_select_rules():
    edges = [("2.2", "1.1"), ("3.3", "1.1"), ("4.4", "2.2"), ("1.1", "2.2"), ("5.5", "6.6"), ("7.7", "8.8")]
    out = select(edges, set(), 10, seed=0)
    papers = [p for pair in out for p in pair]
    assert len(papers) == len(set(papers))                  # mỗi paper tối đa một lần
    assert all(a > b for a, b in out)                       # A mới hơn B
    assert ("1.1", "2.2") not in out                        # A cũ hơn B bị loại
    assert select(edges, {"6.6"}, 10, seed=0) != out or ("5.5", "6.6") not in select(edges, {"6.6"}, 10, seed=0)
    hub = [(f"{i}.0", "0.1") for i in range(20, 40)]       # B in-degree 20 > 10 -> loại
    assert select(hub, set(), 5, seed=0, max_b_indeg=10) == []
    assert select(edges, set(), 10, seed=1) == select(edges, set(), 10, seed=1)   # tái lập
