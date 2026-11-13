import re, time, tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def pack(hits, store, budget):
    ctx, used, seen = [], 0, set()
    for h in hits:
        c = store[h.chunk_id]
        if c["paper_id"] in seen:
            continue                                  # mỗi paper tối đa một lần
        n = len(enc.encode(c["text"]))
        if used + n > budget:
            continue                                  # thử chunk sau, có thể ngắn hơn
        ctx.append(c); used += n; seen.add(c["paper_id"])
    return ctx, used

PROMPT = """Answer the question using ONLY the numbered sources.
Cite every claim with [n]. If the sources are insufficient, reply "Insufficient information."

Sources:
{sources}

Question: {question}
Answer:"""

def generate(question, ctx, client, model, max_tokens=400):
    sources = "\n\n".join(f"[{i}] {c['title']} (arXiv:{c['paper_id']})\n{c['text']}" for i, c in enumerate(ctx, 1))
    t0 = time.time()
    resp = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": PROMPT.format(sources=sources, question=question)}])
    text = resp.choices[0].message.content
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= len(ctx)})
    return {"answer": text, "cited_papers": [ctx[i - 1]["paper_id"] for i in cited],
            "latency_s": time.time() - t0,
            "tokens": {"in": resp.usage.prompt_tokens, "out": resp.usage.completion_tokens}}