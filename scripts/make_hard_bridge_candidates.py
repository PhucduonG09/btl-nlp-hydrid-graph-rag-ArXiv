"""Ứng viên bridge KHÓ cho graph MENTIONS, dựng độc lập với graph retrieval (chỉ dùng chuỗi tiêu đề/abstract).
Hai loại, vì benchmark v2 hiện chỉ có bridge 1 bước qua TÊN RIÊNG (30/30 đường nối là tên có trong tiêu đề B):
  chain2  : A -> B -> C. B giới thiệu e1 (tiêu đề 'e1: ...'), A nhắc e1; C giới thiệu e2, B nhắc e2.
            Câu hỏi mô tả A, đáp án nằm ở C. Cần k>=2 bước; gold = [A, B, C].
  concept : cụm danh từ KHÔNG viết tắt/không tên riêng (vd 'speculative decoding'): cụm p nằm trong tiêu đề B và abstract A
            (không nằm trong tiêu đề A), df hiếm. Graph MENTIONS không bắt được loại này (cần dense/graph khái niệm).
  python scripts/make_hard_bridge_candidates.py [--max-df 8]
Đầu ra: results/analysis/hard_bridge_candidates.csv (cột kind, A_id, B_id, C_id, term, term2, df)"""
import sys, json, csv, re, argparse, collections, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.kg.aliases import GENERIC

STOPW = set("a an the of and or in on for to with by from is are was were be this that we our these which as at it its via using use based towards toward into over under between across how what why when can do does not no".split())
WORD = re.compile(r"[a-z][a-z0-9]+(?:-[a-z0-9]+)*")
BAD = GENERIC | set("language model models large llm llms learning deep neural network networks method methods approach approaches framework frameworks task tasks data dataset datasets benchmark benchmarks evaluation analysis study towards toward novel new efficient effective improving improve enhancing enhance training fine-tuning pre-training generation reasoning understanding detection classification prediction system systems paper results performance".split())


def ngrams(text, ns=(2, 3)):
    w = WORD.findall(text.lower()); out = set()
    for n in ns:
        for i in range(len(w) - n + 1):
            g = w[i:i + n]
            if g[0] in STOPW or g[-1] in STOPW: continue
            out.add(" ".join(g))
    return out


CUE = re.compile(r"\b(?:use[sd]?|using|build(?:s|ing)? on|built on|based on|following|follow(?:s)?|adopt(?:s|ed)?|leverag\w+|employ\w*|inspired by|extend(?:s|ed)?|apply|applies|applied|incorporat\w+|utili[sz]\w+)\b[^.;]{0,80}$", re.I)


def relies_on(abstract, phrase):
    """A có dùng/dựa trên phrase: trong 80 ký tự trước phrase (cùng câu) có từ khóa như use/based on/leverage/extend."""
    a = abstract.lower()
    for m in re.finditer(re.escape(phrase), a):
        pre = a[max(0, m.start() - 100):m.start()]
        pre = re.split(r"[.;]", pre)[-1]
        if CUE.search(pre + "x") or re.search(r"\b(?:use[sd]?|using|based on|leverag\w+|adopt\w*|extend\w*|employ\w*|apply|applies|utili[sz]\w+)\b", pre): return True
    return False


def _month(pid):
    m = re.match(r"(\d{2})(\d{2})\.", pid); return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def not_later(a_id, b_id):
    """B không được ra đời sau A (A không thể dựa vào paper chưa tồn tại)."""
    a, b = _month(a_id), _month(b_id); return a is None or b is None or b <= a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-df", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus", default=str(ROOT / "data/processed/benchmark_v2_corpus.jsonl"))
    ap.add_argument("--mentions", default=str(ROOT / "data/kg/reference_mentions_v2_assisted.json"))
    ap.add_argument("--out", default=str(ROOT / "results/analysis/hard_bridge_candidates.csv")); a = ap.parse_args()
    P = {}
    for l in open(a.corpus, encoding="utf-8"):
        if l.strip(): d = json.loads(l); P[d["paper_id"]] = d
    M = json.load(open(a.mentions, encoding="utf-8")); pe, ep, lab = M["paper_entities"], M["entity_papers"], M["labels"]
    # ---- chain2
    def introduced(b, e):   # tiêu đề B bắt đầu bằng tên thực thể + ':' hoặc '-'
        return bool(re.match(r"\W*" + re.escape(lab.get(e, e)) + r"\s*[:\-–—]", P[b]["title"], re.I))
    intro = {}              # e -> các B giới thiệu e
    for e, ps in ep.items():
        if 2 <= len(ps) <= a.max_df + 4:
            bs = [b for b in ps if introduced(b, e)]
            if len(bs) == 1: intro[e] = bs[0]
    users = collections.defaultdict(list)     # B -> [(e, A)]: A nhắc e (không ở tiêu đề A), B giới thiệu e
    for e, b in intro.items():
        for A in ep[e]:
            if A != b and not_later(A, b) and e not in M["title_entities"].get(A, []): users[b].append((e, A))
    rows = []
    for A_B in [(A, e1, B) for B, lst in users.items() for e1, A in lst]:
        A, e1, B = A_B
        for e2 in pe.get(B, []):
            C = intro.get(e2)
            if C and C not in (A, B) and e2 != e1 and e2 not in M["title_entities"].get(B, []):
                rows.append({"kind": "chain2", "A_id": A, "B_id": B, "C_id": C, "term": lab.get(e1, e1), "term2": lab.get(e2, e2), "df": len(ep[e1])})
    n_chain = len(rows)
    # ---- concept (cụm không viết tắt)
    title_g = {pid: {g for g in ngrams(p["title"]) if not (set(g.split()) <= BAD) and not any(w in BAD for w in g.split()[:1] + g.split()[-1:])} for pid, p in P.items()}
    cand = set().union(*title_g.values()); df = collections.Counter(); where = collections.defaultdict(list)
    for pid, p in P.items():
        for g in ngrams(p["abstract"]) & cand: df[g] += 1; where[g].append(pid)
    for g, pids in where.items():
        if not (2 <= df[g] <= a.max_df): continue
        for B in pids:
            if g not in title_g[B]: continue
            for A in pids:
                if A != B and not_later(A, B) and g not in title_g[A] and relies_on(P[A]["abstract"], g):
                    rows.append({"kind": "concept", "A_id": A, "B_id": B, "C_id": "", "term": g, "term2": "", "df": df[g]})
    rng = random.Random(a.seed); rng.shuffle(rows); rows.sort(key=lambda r: (r["kind"] != "chain2", r["df"], -len(r["term"].split())))   # hiếm và đặc thù lên trước
    keep, usedA, perB = [], set(), collections.Counter()
    for r in rows:   # đa dạng: mỗi A một lần, mỗi B tối đa 2
        if r["A_id"] in usedA or perB[r["B_id"]] >= 2: continue
        usedA.add(r["A_id"]); perB[r["B_id"]] += 1; keep.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "A_id", "B_id", "C_id", "term", "term2", "df"]); w.writeheader(); w.writerows(keep)
    c = collections.Counter(r["kind"] for r in keep)
    print(f"chain2 thô: {n_chain} | tổng thô: {len(rows)} | sau khử trùng: {dict(c)}")
    for k in ("chain2", "concept"):
        print(f"--- mẫu {k}")
        for r in [x for x in keep if x["kind"] == k][:6]:
            print(f"  [{r['term']}{' / ' + r['term2'] if r['term2'] else ''} df={r['df']}] A: {P[r['A_id']]['title'][:60]!r} -> B: {P[r['B_id']]['title'][:60]!r}" + (f" -> C: {P[r['C_id']]['title'][:50]!r}" if r["C_id"] else ""))


if __name__ == "__main__":
    main()
