from src.eval.citation_bridge import accept, cohen_kappa


def test_accept_rules():
    assert accept(("Y", "N"), ("Y", "N")) == "accepted"
    assert accept(("N", "N"), ("N", "Y")) == "rejected"
    assert accept(("Y", "Y"), ("Y", "Y")) == "rejected"          # cả hai nói A tự trả lời được
    assert accept(("Y", "N"), ("N", "N")) == "disagreement"
    assert accept(("Y", "N"), ("Y", "Y")) == "disagreement"
    assert accept(("Y", "N"), ("", "")) == "pending" and accept(("Y", None), ("Y", "N")) == "pending"


def test_kappa():
    assert abs(cohen_kappa(list("YYNN"), list("YYNN")) - 1) < 1e-9
    assert abs(cohen_kappa(list("YYNN"), list("NNYY")) + 1) < 1e-9
    assert cohen_kappa(list("YYYY"), list("YYYY")) is None and cohen_kappa([], []) is None
    assert abs(cohen_kappa(list("YYNN"), list("YNNN")) - 0.5) < 1e-9
