"""Cụm khái niệm (keyphrase) không cần LLM: RAKE-like. Chạy bằng chuỗi từ không phải stopword, rồi lấy n-gram 2..3.
Dùng cho thí nghiệm thăm dò (b-lite): đo trần kết nối của thực thể khái niệm so với MENTIONS (tên riêng/viết tắt)."""
import math, re
from collections import Counter

STOP = set("""a an the of and or in on for to with by from is are was were be been being this that these those which who whom whose as at it its
we our us you your they their them he she his her not no nor but if then than so such can could may might will would shall should do does did done
have has had having into onto over under between among across through during before after above below up down out off again further once here there
when where why how all any both each few more most other some own same too very just also only both either neither whether while about against
via per using use used uses based novel new existing recent paper propose proposes proposed present presents presented introduce introduces show shows
shown demonstrate demonstrates demonstrated study studies approach approaches method methods result results find finds found work works however
thus therefore moreover furthermore specifically particularly significantly improve improves improved achieve achieves achieved outperform outperforms
experiment experiments experimental evaluate evaluation evaluated performance task tasks problem problems various several many different multiple
one two three first second third large small high low better best well compared compare across including include includes provide provides provided
address addresses enable enables allow allows leverage leverages obtain obtains develop develops developed need needs analysis analyze analyzes
framework""".split())
WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def stem(w):
    return w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")) else w


def candidate_phrases(text, nmin=2, nmax=3):
    """Tập cụm n-gram (n trong [nmin,nmax]) nằm trong cùng đoạn, không chứa stopword/số thuần."""
    out = set(); runs = []
    for seg in re.split(r"[.,;:()\[\]!?\"'/]", text.lower()):
        run = []
        for w in WORD.findall(seg):
            if w in STOP or w.isdigit(): runs.append(run); run = []
            else: run.append(stem(w))
        runs.append(run)
    for run in runs:
        for n in range(nmin, nmax + 1):
            for i in range(len(run) - n + 1): out.add(" ".join(run[i:i + n]))
    return out


def paper_phrases(papers):
    return {p["paper_id"]: candidate_phrases(p["title"] + ". " + p["abstract"]) for p in papers}


def top_salient(phrases_by_paper, k):
    """Giữ k cụm có tf-idf (ở đây tf=1 mỗi paper nên = idf; ưu tiên cụm dài hơn khi hòa) cao nhất mỗi paper, rồi tính lại df trên tập đã giữ."""
    df = Counter(p for s in phrases_by_paper.values() for p in s); n = len(phrases_by_paper)
    kept = {}
    for pid, s in phrases_by_paper.items():
        ranked = sorted((p for p in s if df[p] >= 2), key=lambda p: (-math.log(n / df[p]), -len(p.split()), p))
        kept[pid] = set(ranked[:k])
    return kept


def entity_index(phrases_by_paper, min_df=2, max_df=200):
    df = Counter(p for s in phrases_by_paper.values() for p in s)
    return {pid: {p for p in s if min_df <= df[p] <= max_df} for pid, s in phrases_by_paper.items()}
