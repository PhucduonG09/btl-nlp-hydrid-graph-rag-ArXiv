"""Bảng ablation từ kết quả đã có (offline, không API, không mở test). Đọc dev_01/metrics.json (43 cấu hình) và test_01/metrics.json.

Đầu ra: results/analysis/ablation_dev.csv, results/analysis/ablation_tables.md
Mọi số DEV là số trên tập dùng để CHỌN cấu hình (lạc quan); TEST chỉ gồm 5 phương pháp đã khóa, mở một lần.
"""
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/benchmark_v2_assisted"
COLS = [("single", "R@1"), ("single", "MRR@10"), ("single", "nDCG@10"), ("single", "R@10"),
        ("bridge", "All@10"), ("bridge", "Hop2@10"), ("bridge", "R@10"), ("all", "nDCG@10")]
HEAD = ["S R@1", "S MRR@10", "S nDCG@10", "S R@10", "B All@10", "B Hop2@10", "B R@10", "All nDCG@10"]


def family(name):
    """Phân loại cấu hình theo thành phần. Trả (nhóm, mô tả)."""
    if name.startswith("bm25_"):
        return "A. BM25 (tham số k1,b)", name
    if name == "dense":
        return "B. Dense (bge-base-en-v1.5)", name
    if re.fullmatch(r"rrf_k\d+", name):
        return "C. BM25+dense qua RRF (tham số k)", name
    if name.endswith("text_expansion"):
        return "D. Đối chứng mở rộng văn bản", name
    m = re.fullmatch(r"(rrf|linked)_(k[123]|ppr)_(rrf_w[\d.]+|weighted)", name)
    if m:
        seed = "seed từ top RRF (không liên kết thực thể)" if m.group(1) == "rrf" else "seed từ thực thể liên kết trong câu hỏi"
        return f"E. Graph MENTIONS, {seed}", name
    raise ValueError(f"cấu hình chưa phân loại: {name}")


def row(metrics):
    return [metrics[g][k] for g, k in COLS]


def fmt(v):
    return f"{v:.3f}"


def main():
    dev = json.loads((BASE / "dev_01/metrics.json").read_text(encoding="utf-8"))
    test = json.loads((BASE / "test_01/metrics.json").read_text(encoding="utf-8"))
    selected = json.loads((BASE / "dev_01/selected_config.json").read_text(encoding="utf-8"))
    chosen = {selected["bm25"]: "BM25 chọn", selected["rrf"]: "RRF chọn", selected["graph"]: "Graph chọn"}
    out = ROOT / "results/analysis"
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for name, m in dev.items():
        g, _ = family(name)
        groups.setdefault(g, []).append((name, m))
    with (out / "ablation_dev.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "group", "selected"] + [f"{g}_{k}" for g, k in COLS])
        for g in sorted(groups):
            for name, m in groups[g]:
                w.writerow([name, g, chosen.get(name, "")] + [round(x, 4) for x in row(m)])
    lines = ["# Bảng ablation (tự sinh bởi scripts/make_ablation_table.py)", "",
             "Mọi số DEV là trên tập dùng để chọn cấu hình (lạc quan). S = single (n=91), B = bridge (n=31). In đậm = cấu hình đã chọn trên DEV.", ""]
    # Bảng 1: bỏ từng thành phần khỏi hệ thống đã chọn
    sel_graph, sel_rrf, sel_bm25 = selected["graph"], selected["rrf"], selected["bm25"]
    k1 = sel_graph.replace("linked_", "rrf_")
    wt = sel_graph.rsplit("_rrf_w", 1)[0] + "_weighted"
    steps = [(f"Hệ thống đầy đủ: {sel_graph}", sel_graph),
             ("Bỏ liên kết thực thể (seed từ top RRF)", k1),
             ("Bỏ graph (chỉ BM25+dense qua RRF)", sel_rrf),
             ("Bỏ dense (chỉ BM25)", sel_bm25),
             ("Bỏ BM25 (chỉ dense)", "dense"),
             ("Thay graph bằng mở rộng văn bản (cùng seed)", "linked_text_expansion"),
             ("Fusion trọng số điểm thay RRF (linked_k1)", wt.replace("rrf_", "linked_") if wt.startswith("rrf_") else wt),
             ("Mở rộng 2 bước thay vì 1", sel_graph.replace("_k1_", "_k2_")),
             ("Mở rộng 3 bước thay vì 1", sel_graph.replace("_k1_", "_k3_"))]
    lines += ["## 1. Bỏ từng thành phần khỏi hệ thống đã chọn (DEV)", "", "| Biến thể | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
    for label, key in steps:
        if key not in dev:
            raise KeyError(f"thiếu cấu hình {key} cho dòng '{label}'")
        lines.append(f"| {label} (`{key}`) | " + " | ".join(fmt(x) for x in row(dev[key])) + " |")
    lines += ["", "## 2. Toàn bộ 43 cấu hình DEV theo nhóm", ""]
    for g in sorted(groups):
        lines += [f"### {g}", "", "| Cấu hình | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
        for name, m in groups[g]:
            tag = f"**{name}**" if name in chosen else name
            lines.append(f"| {tag} | " + " | ".join(fmt(x) for x in row(m)) + " |")
        lines.append("")
    lines += ["## 3. TEST (28 câu: 14 single + 14 bridge; mở một lần; 5 phương pháp đã khóa)", "",
              "| Phương pháp | " + " | ".join(HEAD) + " |", "|---|" + "---:|" * len(HEAD)]
    for name, m in test.items():
        lines.append(f"| {name} | " + " | ".join(fmt(x) for x in row(m)) + " |")
    (out / "ablation_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = sum(len(v) for v in groups.values())
    assert n == len(dev) == 43, n
    print(f"{n} cấu hình, {len(groups)} nhóm. Đã ghi results/analysis/ablation_dev.csv và ablation_tables.md")


if __name__ == "__main__":
    main()
