"""Xây KG (NetworkX) từ metadata + kết quả trích xuất, ghi graphml, triples và thống kê.
  python scripts/build_kg.py --extract data/kg/pilot_extract_v3.jsonl --tag v3"""
import sys, json, argparse
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import networkx as nx
from src.kg.aliases import canon, merge_acronyms, GENERIC

ap = argparse.ArgumentParser()
ap.add_argument("--papers", default=str(ROOT / "data/processed/pilot_papers.jsonl"))
ap.add_argument("--extract", required=True)
ap.add_argument("--tag", default="v3")
ap.add_argument("--drop-generic", action="store_true", default=True)
a = ap.parse_args()

papers = {}
for l in open(a.papers, encoding="utf-8"):
    if l.strip():
        d = json.loads(l); papers[d["paper_id"]] = d
ext = {}
for l in open(a.extract, encoding="utf-8"):
    if l.strip():
        d = json.loads(l); ext[d["paper_id"]] = d

# 1) gom triple, chuẩn hóa tên, gộp viết tắt ↔ tên đầy đủ
raw = []
for pid, d in ext.items():
    for r in d.get("relations", []):
        c = canon(r["target"])
        if not c or (a.drop_generic and r["relation"] == "USES" and c in GENERIC):
            continue
        raw.append((pid, r["relation"], c, r.get("target_type", "Other")))
counts = Counter(c for _, _, c, _ in raw)
amap = merge_acronyms(counts)
triples = sorted({(p, rel, amap.get(c, c), t) for p, rel, c, t in raw})

# 2) dựng đồ thị
G = nx.MultiDiGraph()
for pid in ext:
    p = papers.get(pid)
    if not p:
        continue
    G.add_node(f"paper:{pid}", type="Paper", label=p["title"])
    for au in p.get("authors", []):
        G.add_node(f"author:{au}", type="Author", label=au)
        G.add_edge(f"paper:{pid}", f"author:{au}", rel="written_by")
    for cat in p.get("categories", []):
        G.add_node(f"cat:{cat}", type="Category", label=cat)
        G.add_edge(f"paper:{pid}", f"cat:{cat}", rel="belongs_to")
ttypes = defaultdict(Counter)
for p, rel, c, t in triples:
    ttypes[c][t] += 1
for p, rel, c, t in triples:
    G.add_node(f"concept:{c}", type="Concept", label=c, ctype=ttypes[c].most_common(1)[0][0])
    G.add_edge(f"paper:{p}", f"concept:{c}", rel=rel)

out_dir = ROOT / "data" / "kg"
nx.write_graphml(G, out_dir / f"kg_{a.tag}.graphml")
with open(out_dir / f"triples_{a.tag}.jsonl", "w", encoding="utf-8") as f:
    for p, rel, c, t in triples:
        f.write(json.dumps({"paper_id": p, "relation": rel, "concept": c, "concept_type": t}, ensure_ascii=False) + "\n")

# 3) thống kê (đưa vào báo cáo)
paper_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "Paper"]
concept_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "Concept"]
papers_per_concept = Counter()
for p, rel, c, t in triples:
    papers_per_concept[c] += 0
cp = defaultdict(set)
for p, rel, c, t in triples:
    cp[c].add(p)
multi = {c: ps for c, ps in cp.items() if len(ps) >= 2}
H = nx.Graph()                      # chiếu paper–concept để đo thành phần liên thông
H.add_nodes_from(paper_nodes)
for p, rel, c, t in triples:
    H.add_edge(f"paper:{p}", f"concept:{c}")
comps = sorted((len(x) for x in nx.connected_components(H)), reverse=True)
stats = {
    "tag": a.tag, "n_papers_with_extraction": len(ext), "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
    "nodes_by_type": dict(Counter(d["type"] for _, d in G.nodes(data=True))),
    "edges_by_rel": dict(Counter(d["rel"] for _, _, d in G.edges(data=True))),
    "papers_without_relation": sum(1 for p in ext if not any(t[0] == p for t in triples)),
    "concepts_in_ge2_papers": len(multi), "concepts_in_ge2_papers_pct": round(100 * len(multi) / max(len(concept_nodes), 1), 1),
    "top_concepts": [(c, len(ps)) for c, ps in sorted(cp.items(), key=lambda x: -len(x[1]))[:15]],
    "largest_component_size": comps[0] if comps else 0, "n_components": len(comps),
    "extraction_json_errors": sum("error" in d for d in ext.values()),
}
(out_dir / f"kg_stats_{a.tag}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(stats, ensure_ascii=False, indent=2))
