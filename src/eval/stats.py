import numpy as np

def paired_bootstrap_ci(a, b, n=10000, seed=42):
    """a, b: điểm từng câu của hệ thống A và B, cùng thứ tự câu hỏi. Trả về CI 95% của mean(B - A)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    return float(b.mean() - a.mean()), tuple(np.percentile(diffs, [2.5, 97.5]))

def paired_permutation_p(a, b, n=10000, seed=42):
    """Sign-flip permutation test hai phía cho chênh lệch trung bình."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= abs(d.mean())) + 1) / (n + 1))