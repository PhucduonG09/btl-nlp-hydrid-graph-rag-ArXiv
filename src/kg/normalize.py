import re
from collections import Counter

ACRO = re.compile(r"([A-Z][A-Za-z\-]+(?:\s+[A-Za-z\-]+){1,6})\s*\(([A-Z][A-Za-z0-9\-]{1,12})\)")

def key(s: str) -> str:                       # tầng 1: chuỗi
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def acronym_map(abstracts):                     # tầng 2: viết tắt định nghĩa trong chính abstract
    votes = {}
    for text in abstracts:
        for long, short in ACRO.findall(text):
            votes.setdefault(key(short), Counter())[key(long)] += 1
    return {s: c.most_common(1)[0][0] for s, c in votes.items()}

def canonical(name: str, acro: dict) -> str:
    k = key(name)
    return acro.get(k, k)