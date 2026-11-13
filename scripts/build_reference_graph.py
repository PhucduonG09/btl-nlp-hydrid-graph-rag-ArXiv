"""Build a full-corpus MENTIONS retrieval control without reading any QA files."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.kg.mentions import build_mention_graph

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',default='data/processed/bridge_corpus_v1.jsonl')
    ap.add_argument('--output',default='data/kg/reference_mentions_v1.json');args=ap.parse_args()
    path=ROOT/args.corpus
    papers=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    graph=build_mention_graph(papers)
    graph['corpus_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    graph['build_source_sha256']=hashlib.sha256((ROOT/'src/kg/mentions.py').read_bytes()).hexdigest()
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(graph,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(graph['stats'],indent=2))
if __name__=='__main__':main()
