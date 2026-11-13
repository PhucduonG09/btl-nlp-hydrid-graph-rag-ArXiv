import pytest
from src.generation.ordering import reorder, MODES

R = list("abcdefghij")


def test_sandwich_puts_best_at_both_edges():
    out = reorder(R, "sandwich")
    assert out == list("acegijhfdb")
    assert out[0] == "a" and out[-1] == "b"          # hạng 1 và 2 ở hai rìa
    assert set(out[4:6]) == {"i", "j"}               # hạng thấp nhất ở giữa


@pytest.mark.parametrize("mode", ["rank", "reverse", "sandwich"])
def test_orderings_are_permutations(mode):
    assert sorted(reorder(R, mode)) == R


def test_reverse_and_top5_and_rank():
    assert reorder(R, "reverse") == R[::-1]
    assert reorder(R, "top5") == list("abcde")
    assert reorder(R, "rank") == R


def test_edge_cases():
    assert reorder(["a"], "sandwich") == ["a"] and reorder(["a", "b"], "sandwich") == ["a", "b"]
    with pytest.raises(ValueError):
        reorder(["a", "a"], "rank")
    with pytest.raises(ValueError):
        reorder(R, "nope")
    assert set(MODES) == {"rank", "reverse", "sandwich", "top5"}
