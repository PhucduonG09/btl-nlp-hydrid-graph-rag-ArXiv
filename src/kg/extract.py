import os, json
from typing import Literal
from pydantic import BaseModel, ValidationError
from openai import OpenAI

class Relation(BaseModel):
    relation: Literal["PROPOSES", "USES", "ADDRESSES", "EVALUATES_ON"]
    target: str
    target_type: Literal["Method", "Task", "Dataset"]
    evidence: str                       # cụm trích NGUYÊN VĂN từ title/abstract

class Extraction(BaseModel):
    relations: list[Relation]

SYSTEM = """You extract a knowledge graph from a scientific abstract.
Return JSON: {"relations": [{"relation", "target", "target_type", "evidence"}]}.
Relations (subject is always the paper):
- PROPOSES: the paper introduces a NEW method/model/technique (usually named by the authors).
- USES: the paper uses an EXISTING method as a component or backbone. Do NOT use it for methods only compared against.
- ADDRESSES: the task/problem the paper tackles.
- EVALUATES_ON: a named dataset or benchmark used for evaluation.
Rules: use the shortest standard name for target (e.g. "BERT", not "the BERT model").
"evidence" must be copied verbatim from the text. Do not infer anything not stated.
If nothing applies, return {"relations": []}."""

client = OpenAI(base_url=os.environ.get("LLM_BASE_URL"), api_key=os.environ.get("LLM_API_KEY", "local"))

def extract(paper: dict, model: str) -> dict:
    text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        response_format={"type": "json_object"},   # bỏ dòng này nếu endpoint không hỗ trợ
    )
    raw = resp.choices[0].message.content
    out = {"paper_id": paper["paper_id"], "extractor": model,
           "usage": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}}
    try:
        ex = Extraction.model_validate_json(raw)
    except ValidationError:
        return {**out, "relations": [], "error": "invalid_json", "raw": raw}
    src = text.lower()
    kept = [r.model_dump() for r in ex.relations if r.evidence.lower() in src]
    out["relations"] = kept
    out["dropped_no_evidence"] = len(ex.relations) - len(kept)   # báo cáo con số này
    return out