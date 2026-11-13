"""Trích xuất quan hệ (paper → khái niệm) bằng LLM, chịu lỗi: mỗi quan hệ được kiểm tra riêng."""
import json, re, threading
from pathlib import Path
from src.common.llm import chat, run_parallel
from src.kg.prompts import PROMPTS
from src.kg.evidence import locate_evidence

VALID_REL = {"PROPOSES", "USES", "ADDRESSES", "EVALUATES_ON"}
TYPE_MAP = {"method": "Method", "model": "Method", "technique": "Method", "algorithm": "Method",
            "framework": "Method", "tool": "Method", "approach": "Method",
            "task": "Task", "problem": "Task",
            "dataset": "Dataset", "benchmark": "Dataset", "resource": "Dataset", "corpus": "Dataset"}


def _ws(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def parse_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(raw[i:j + 1])
    except json.JSONDecodeError:
        return None


def extract_one(paper, prompt_version="v3", model=None, json_mode=False, max_tokens=None):
    text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
    raw, usage = chat([{"role": "system", "content": PROMPTS[prompt_version]},
                       {"role": "user", "content": text}], model=model, json_mode=json_mode, max_tokens=max_tokens)
    out = {"paper_id": paper["paper_id"], "prompt": prompt_version, "usage": usage}
    data = parse_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
        return {**out, "relations": [], "error": "invalid_json", "raw": raw[:2000]}
    src, kept, no_ev, invalid, seen = _ws(text), [], 0, 0, set()
    for r in data["relations"]:
        try:
            rel = str(r["relation"]).strip().upper()
            tgt = str(r["target"]).strip()
            ev = str(r.get("evidence", "")).strip()
        except (KeyError, TypeError, AttributeError):
            invalid += 1
            continue
        if rel not in VALID_REL or not tgt:
            invalid += 1
            continue
        span=locate_evidence(text,ev)
        if span is None:
            no_ev += 1
            continue
        if (rel, _ws(tgt)) in seen:
            continue
        seen.add((rel, _ws(tgt)))
        ttype = TYPE_MAP.get(str(r.get("target_type", "")).strip().lower(), "Other")
        kept.append({"relation": rel, "target": tgt, "target_type": ttype, "evidence": span.text,
                     "evidence_start":span.start,"evidence_end":span.end,"evidence_match":span.match_kind})
    return {**out, "relations": kept, "dropped_no_evidence": no_ev, "n_invalid_items": invalid}


def load_done(path):
    p = Path(path)
    if not p.exists():
        return {}
    done = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            done[d["paper_id"]] = d
        except Exception:
            continue
    return done


def run_extraction(papers, out_path, prompt_version="v3", model=None, workers=6, json_mode=False, max_tokens=None):
    """Resume được: paper đã có trong out_path (và không lỗi) sẽ không gọi lại."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    done = {k: v for k, v in load_done(out_path).items() if "error" not in v}
    todo = [p for p in papers if p["paper_id"] not in done]
    lock = threading.Lock()

    def work(p):
        r = extract_one(p, prompt_version, model, json_mode, max_tokens)
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r
    results, errors = run_parallel(work, todo, workers=workers, desc=f"extract {prompt_version}")
    final = load_done(out_path)          # lấy bản ghi mới nhất cho mỗi paper
    return [final[p["paper_id"]] for p in papers if p["paper_id"] in final], errors
