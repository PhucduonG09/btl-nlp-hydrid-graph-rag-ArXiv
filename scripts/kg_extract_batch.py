"""Trích xuất quan hệ, gộp B paper mỗi lần gọi để chia nhỏ phần token cố định (~3k/lần gọi theo usage pilot).
  python scripts/kg_extract_batch.py --input data/eval/dev_extract.jsonl --out data/kg/dev_extract_out_v3s_b4.jsonl --prompt v3s --batch 4
Cùng định dạng đầu ra với kg_extract.py (chấm bằng eval_extraction*.py được). usage mỗi paper = usage lượt gọi / B.
Evidence được kiểm tra CHỈ trên văn bản của chính paper đó → phát hiện rò rỉ chéo giữa các paper.
Resume được. Paper nào thiếu/hỏng trong lượt gộp sẽ được gọi lại riêng lẻ."""
import sys, json, argparse, time, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.llm import chat, run_parallel, cost_summary
from src.kg.prompts import PROMPTS
from src.kg.llm_extract import parse_json, _ws, VALID_REL, TYPE_MAP, load_done, extract_one

ADDENDUM = ('\n\nYou will receive several papers, each starting with "### PAPER <id>". Treat every paper independently; '
            '"evidence" must come from that paper only. Return ONLY JSON: {"papers": [{"id": "<id>", "relations": [...]}]} '
            "with exactly one entry per paper id, in the same order.")


def validate(paper, rels, usage):
    text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
    src, kept, no_ev, invalid, seen = _ws(text), [], 0, 0, set()
    for r in rels if isinstance(rels, list) else []:
        try:
            rel, tgt, ev = str(r["relation"]).strip().upper(), str(r["target"]).strip(), str(r.get("evidence", "")).strip()
        except (KeyError, TypeError, AttributeError):
            invalid += 1; continue
        if rel not in VALID_REL or not tgt: invalid += 1; continue
        if not ev or _ws(ev) not in src: no_ev += 1; continue
        if (rel, _ws(tgt)) in seen: continue
        seen.add((rel, _ws(tgt)))
        kept.append({"relation": rel, "target": tgt, "target_type": TYPE_MAP.get(str(r.get("target_type", "")).strip().lower(), "Other"),
                     "evidence": ev})
    return {"paper_id": paper["paper_id"], "usage": usage, "relations": kept, "dropped_no_evidence": no_ev, "n_invalid_items": invalid}


def extract_group(group, prompt, model, max_tokens):
    body = "\n\n".join(f"### PAPER {p['paper_id']}\nTitle: {p['title']}\nAbstract: {p['abstract']}" for p in group)
    raw, u = chat([{"role": "system", "content": PROMPTS[prompt] + ADDENDUM}, {"role": "user", "content": body}],
                  model=model, max_tokens=max_tokens)
    data = parse_json(raw)
    by_id = {}
    if isinstance(data, dict) and isinstance(data.get("papers"), list):
        for d in data["papers"]:
            if isinstance(d, dict) and "id" in d: by_id[str(d["id"]).strip()] = d.get("relations", [])
    share = {**u, "in": u["in"] / len(group), "out": u["out"] / len(group), "batch": len(group)}
    outs, redo = [], []
    for p in group:
        if p["paper_id"] in by_id: outs.append({**validate(p, by_id[p["paper_id"]], share), "prompt": prompt + f"+b{len(group)}"})
        else: redo.append(p)
    for p in redo:                                   # thiếu trong lượt gộp → gọi riêng
        outs.append(extract_one(p, prompt, model))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="v3s"); ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    papers = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    if a.limit: papers = papers[:a.limit]
    done = {k for k, v in load_done(a.out).items() if "error" not in v}
    todo = [p for p in papers if p["paper_id"] not in done]
    groups = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); lock = threading.Lock(); t0 = time.time()

    def work(g):
        res = extract_group(g, a.prompt, a.model, a.max_tokens)
        with lock, open(a.out, "a", encoding="utf-8") as f:
            for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return res
    results, errors = run_parallel(work, groups, workers=a.workers, desc=f"extract {a.prompt} b{a.batch}")
    outs = [r for g in results if g for r in g]
    print(f"\n{len(outs)}/{len(todo)} paper mới | {sum(len(o['relations']) for o in outs)} quan hệ | {time.time()-t0:.0f}s | lỗi nhóm: {len(errors)}")
    print("Chi phí lượt này:", cost_summary([o["usage"] for o in outs if o["usage"].get("cached") is False]))
    print("Thiếu evidence bị loại:", sum(o.get("dropped_no_evidence", 0) for o in outs), "(tăng bất thường so với chạy lẻ = dấu hiệu rò rỉ chéo)")


if __name__ == "__main__":
    main()
