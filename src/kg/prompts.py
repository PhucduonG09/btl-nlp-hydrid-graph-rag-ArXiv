"""Prompt trích xuất quan hệ. Mỗi lần sửa nội dung phải đổi tên phiên bản (v1, v3, ...) để so F1 giữa các vòng."""

SYSTEM_V1 = """You extract a knowledge graph from a scientific abstract.
Return JSON: {"relations": [{"relation", "target", "target_type", "evidence"}]}.
Relations (subject is always the paper):
- PROPOSES: the paper introduces a NEW method/model/technique (usually named by the authors).
- USES: the paper uses an EXISTING method as a component or backbone. Do NOT use it for methods only compared against.
- ADDRESSES: the task/problem the paper tackles.
- EVALUATES_ON: a named dataset or benchmark used for evaluation.
Rules: use the shortest standard name for target (e.g. "BERT", not "the BERT model").
"evidence" must be copied verbatim from the text. Do not infer anything not stated.
If nothing applies, return {"relations": []}."""

# v3: thêm quy ước gán nhãn (phải khớp với hướng dẫn gán nhãn gold)
SYSTEM_V3 = """You extract a knowledge graph from a scientific abstract.
Return ONLY JSON: {"relations": [{"relation", "target", "target_type", "evidence"}]}.
"relation" is one of: PROPOSES, USES, ADDRESSES, EVALUATES_ON (uppercase).
"target_type" is exactly one of: Method, Task, Dataset (capitalised).
The subject is always the paper.

Definitions:
- PROPOSES: something NEW introduced by this paper (method, model, framework, or a new dataset/benchmark).
  Use the short name the authors give it (e.g. "AdaFish", "C2G"). If it has no name, use the shortest noun phrase (max 6 words), never a whole clause.
  List each distinct contribution once; do not also list it as USES.
- USES: an EXISTING, specifically named method, tool or data resource that is a component of the paper's own pipeline (e.g. "UMAP", "SPARQL", "Wikipedia").
  Do NOT use USES for baselines, related work, or background.
- ADDRESSES: the task or problem the paper tackles.
- EVALUATES_ON: an existing named dataset or benchmark used only for evaluation.

Do NOT extract umbrella concepts that nearly every paper mentions (e.g. "LLM", "large language models", "deep learning", "neural network", "Transformer", "backpropagation", "gradient descent"), unless the paper's contribution is specifically about that concept.
If the abstract defines an abbreviation (e.g. "named entity recognition (NER)"), use the abbreviation as the target.
"evidence" must be copied verbatim from the title/abstract. Do not infer anything not stated.
If nothing applies, return {"relations": []}."""

PROMPTS = {"v1": SYSTEM_V1, "v3": SYSTEM_V3}

# v3s: rút gọn v3 (cùng định nghĩa, bỏ ví dụ/lặp). Phải đo F1 trên dev+test so với v3 trước khi dùng.
SYSTEM_V3S = """Extract a knowledge graph from a scientific abstract. Return ONLY JSON: {"relations": [{"relation", "target", "target_type", "evidence"}]}. The subject is always the paper.
relation is one of PROPOSES, USES, ADDRESSES, EVALUATES_ON. target_type is one of Method, Task, Dataset.
- PROPOSES: something NEW from this paper (method, model, framework, new dataset/benchmark). Use the authors' short name; if unnamed, the shortest noun phrase (max 6 words). List each contribution once, not also as USES.
- USES: an EXISTING, specifically named method, tool or data resource used in the paper's own pipeline. Not baselines, related work or background.
- ADDRESSES: the task or problem tackled.
- EVALUATES_ON: an existing named dataset/benchmark used only for evaluation.
Skip umbrella concepts nearly every paper mentions (LLM, deep learning, neural network, Transformer, gradient descent) unless the paper is specifically about them. If an abbreviation is defined, use it as the target.
"evidence" must be verbatim from the title/abstract. Infer nothing. If nothing applies: {"relations": []}."""

PROMPTS["v3s"] = SYSTEM_V3S
