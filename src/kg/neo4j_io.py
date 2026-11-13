import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                              auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))

CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Task) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
]

PAPERS = """
UNWIND $rows AS r
MERGE (p:Paper {paper_id: r.paper_id})
SET p.title = r.title, p.year = toInteger(substring(r.date, 0, 4)), p.primary_category = r.primary_category
WITH p, r
UNWIND r.authors AS name
MERGE (a:Author {name: name})
MERGE (p)-[:AUTHORED_BY]->(a)
"""

# Kiểu cạnh không truyền tham số được trong Cypher -> sinh query từ whitelist
REL_TEMPLATE = """
UNWIND $rows AS r
MATCH (p:Paper {{paper_id: r.paper_id}})
MERGE (t:{label} {{name: r.target}})
MERGE (p)-[e:{rel}]->(t)
SET e.evidence = r.evidence, e.extractor = r.extractor
"""
ALLOWED = {"PROPOSES": "Method", "USES": "Method", "ADDRESSES": "Task", "EVALUATES_ON": "Dataset"}

def run_batched(query, rows, size=1000):
    with driver.session() as s:
        for i in range(0, len(rows), size):
            s.run(query, rows=rows[i:i + size])

def load(papers, relations):
    with driver.session() as s:
        for c in CONSTRAINTS:
            s.run(c)
    run_batched(PAPERS, papers)
    for rel, label in ALLOWED.items():
        rows = [r for r in relations if r["relation"] == rel]
        run_batched(REL_TEMPLATE.format(label=label, rel=rel), rows)