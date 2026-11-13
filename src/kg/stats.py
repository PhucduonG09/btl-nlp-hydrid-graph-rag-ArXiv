import networkx as nx
from collections import Counter

def graph_stats(G: nx.Graph) -> dict:
    types = Counter(d["type"] for _, d in G.nodes(data=True))
    methods = [n for n, d in G.nodes(data=True) if d["type"] == "Method"]
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    return {
        "nodes_by_type": dict(types),
        "edges": G.number_of_edges(),
        "methods_in_2plus_papers": sum(G.degree(m) >= 2 for m in methods) / max(len(methods), 1),
        "largest_component_share": comps[0] / G.number_of_nodes() if comps else 0,
        "n_components": len(comps),
    }