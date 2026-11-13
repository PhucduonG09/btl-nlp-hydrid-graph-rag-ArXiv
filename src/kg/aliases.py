"""Chuẩn hóa tên thực thể (entity normalization) dùng chung cho đánh giá trích xuất và xây KG."""
import re

# Khái niệm quá chung: gần như paper nào cũng nhắc. Chỉ dùng để lọc quan hệ USES (tùy chọn).
GENERIC = {
    "llm", "llms", "large language model", "large language models", "language model", "language models",
    "deep learning", "machine learning", "neural network", "neural networks", "transformer", "transformers",
    "backpropagation", "gradient descent", "attention", "ai", "nlp",
}

# Bảng alias bổ sung tay (khóa và giá trị đều ở dạng đã norm). Thêm dần khi đọc lỗi.
ALIAS = {
    "rag": "retrieval augmented generation",
    "rlhf": "reinforcement learning from human feedback",
    "lora": "low rank adaptation",
    "cot": "chain of thought",
    "gnn": "graph neural network",
    "gnns": "graph neural network",
    "kg": "knowledge graph",
    "kgs": "knowledge graph",
}

_STOPW = {"of", "for", "the", "and", "with", "in", "on", "to", "a", "an", "from"}


def norm(s: str) -> str:
    text=(s or "").lower().replace('++',' plus plus ')
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canon(s: str) -> str:
    base = re.sub(r"\s*\(.*?\)", "", s or "").strip()      # "Large Language Models (LLMs)" -> "Large Language Models"
    n = norm(base or s)
    n = re.sub(r"^(the|a|an) ", "", n)
    return ALIAS.get(n, n)


def _initials(words, skip_stop):
    return "".join(w[0] for w in words if not (skip_stop and w in _STOPW))


def is_acronym(short: str, long: str) -> bool:
    """'ner' ~ 'named entity recognition'; 'llms' ~ 'large language models'."""
    s, l = norm(short), norm(long)
    if " " in s or not (2 <= len(s) <= 8) or " " not in l:
        return False
    words = l.split()
    cands = {_initials(words, False), _initials(words, True)}
    return s in cands or s.rstrip("s") in cands


def match(a: str, b: str, loose: bool = False) -> bool:
    """So khớp hai tên thực thể. Chặt: bằng nhau sau canon. Lỏng: chứa nhau hoặc là viết tắt của nhau."""
    ca, cb = canon(a), canon(b)
    if ca == cb:
        return True
    if not loose:
        return False
    # Neutral presentation suffixes are safe; substring matching is not:
    # ToM is not 'automatic', GPT-4V is not GPT-4, and v2 is not v1.
    neutral={"model","method","algorithm","framework","dataset","benchmark","system"}
    def unadorn(name):
        words=name.split()
        while len(words)>1 and words[-1] in neutral:words.pop()
        return " ".join(words)
    if unadorn(ca)==unadorn(cb):return True
    return is_acronym(ca, cb) or is_acronym(cb, ca)


def merge_acronyms(counts: dict) -> dict:
    """counts: {tên_canon: số lần}. Trả mapping gộp viết tắt về dạng xuất hiện nhiều hơn."""
    names = list(counts)
    longs = [n for n in names if " " in n]
    mapping = {}
    for n in names:
        if " " in n:
            continue
        candidates=[l for l in longs if is_acronym(n,l)]
        if len(candidates)==1:
            l=candidates[0]
            mapping[n]=l if counts[l]>=counts[n] else n
            mapping[l]=mapping[n]
    return mapping
