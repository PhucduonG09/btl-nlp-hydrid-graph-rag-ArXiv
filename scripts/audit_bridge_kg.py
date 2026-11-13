"""Audit existing KG coverage for frozen bridge DEV QA without changing the KG.

Only existing GraphML edges count as paths. Gold IDs/entities are used to audit
coverage, never to fabricate edges. A gold-conditioned diagnostic extraction
input is prepared separately and must not be treated as a full-corpus KG.
"""
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter,defaultdict,deque
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.kg.aliases import canon,match

def load(path):
    with path.open(encoding='utf-8') as f:return [json.loads(l) for l in f if l.strip()]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def distance(adjacency,start,end):
    if start not in adjacency or end not in adjacency:return None
    queue=deque([(start,0)]);seen={start}
    while queue:
        node,d=queue.popleft()
        if node==end:return d
        for neighbor in adjacency[node]-seen:
            seen.add(neighbor);queue.append((neighbor,d+1))
    return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',default='v1');ap.add_argument('--tag',default='v3')
    args=ap.parse_args()
    dataset=ROOT/'data/eval'/f'bridge_{args.version}'
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    qa=load(dataset/'qa_dev.jsonl')
    corpus_path=ROOT/manifest['corpus'];assert sha(corpus_path)==manifest['corpus_sha256']
    corpus={p['paper_id']:p for p in load(corpus_path)}
    graph_path=ROOT/'data/kg'/f'kg_{args.tag}.graphml';graph_hash=sha(graph_path)
    ns={'g':'http://graphml.graphdrawing.org/xmlns'}
    tree=ET.parse(graph_path);root=tree.getroot()
    keys={k.attrib['id']:k.attrib['attr.name'] for k in root.findall('g:key',ns)}
    graph=root.find('g:graph',ns)
    nodes={n.attrib['id']:{keys[d.attrib['key']]:d.text or '' for d in n.findall('g:data',ns)} for n in graph.findall('g:node',ns)}
    adjacency=defaultdict(set);semantic=defaultdict(set);paper_concepts=defaultdict(lambda:defaultdict(set));edges=[]
    for n in nodes:adjacency[n];semantic[n]
    for edge in graph.findall('g:edge',ns):
        a,b=edge.attrib['source'],edge.attrib['target']
        data={keys[d.attrib['key']]:d.text or '' for d in edge.findall('g:data',ns)}
        rel=data.get('rel','');edges.append((a,b,rel))
        adjacency[a].add(b);adjacency[b].add(a)
        if {nodes.get(a,{}).get('type'),nodes.get(b,{}).get('type')}=={'Paper','Concept'}:
            semantic[a].add(b);semantic[b].add(a)
            p,c=(a,b) if nodes.get(a,{}).get('type')=='Paper' else (b,a)
            paper_concepts[p][c].add(rel)
    concepts={n:data for n,data in nodes.items() if data.get('type')=='Concept'}
    strict_index=defaultdict(list)
    for n,data in concepts.items():strict_index[canon(data.get('label',n.removeprefix('concept:')))].append(n)
    available=defaultdict(list)
    for path in sorted((ROOT/'data/kg').glob('*.jsonl')):
        if path.name.startswith('triples_'):continue
        for item in load(path):
            if 'paper_id' in item and 'relations' in item and 'error' not in item:
                available[item['paper_id']].append(dict(file=str(path.relative_to(ROOT)),n_relations=len(item['relations'])))
    # Descriptive relationship categories from reviewed DEV questions, not edges.
    special={24:'REFERENCES',25:'COMPARES_TO',26:'PREDECESSOR',31:'COMPATIBLE_WITH',32:'COMPARES_TO',
             33:'SHARED_DATASET',34:'DEFENDS_AGAINST',44:'COMPARES_TO',46:'ANALYZES',48:'COMPARES_TO',49:'SIBLING_PROJECT',50:'COMPARES_TO'}
    rows=[]
    for q in qa:
        a,b=(f"paper:{q[k]}" for k in ('hop1_id','hop2_id'))
        exact=strict_index.get(canon(q['bridge_entity']),[])
        # Loose candidates are diagnostic only: never count them as a confirmed link.
        candidates=[n for n,d in concepts.items() if n not in exact and match(q['bridge_entity'],d.get('label',''),loose=True)]
        common=set(paper_concepts[a])&set(paper_concepts[b])
        bridge_common=common&set(exact)
        strict_paths=[c for c in bridge_common if paper_concepts[a][c]&{'USES','EVALUATES_ON'} and 'PROPOSES' in paper_concepts[b][c]]
        a_present=nodes.get(a,{}).get('type')=='Paper';b_present=nodes.get(b,{}).get('type')=='Paper'
        sem_distance=distance(semantic,a,b);full_distance=distance(adjacency,a,b)
        relation=special.get(q['source_row'],'USES_OR_EVALUATES_ON') if q['source']=='bridge_drafts2.xlsx' else ('SHARED_CONCEPT' if q['source']=='bridge_drafts.xlsx' and q['source_row']==1 else 'USES_OR_EVALUATES_ON')
        if not a_present or not b_present:
            reason='MISSING_BOTH_PAPERS' if not a_present and not b_present else 'MISSING_A' if not a_present else 'MISSING_B'
            missing=[pid for pid,present in [(q['hop1_id'],a_present),(q['hop2_id'],b_present)] if not present]
            missing_extraction=[pid for pid in missing if pid not in available]
            action='Trích xuất paper thiếu: '+', '.join(missing_extraction) if missing_extraction else 'Có trích xuất lưu sẵn; cần kiểm tra rồi rebuild KG.'
        elif not exact:
            reason='BRIDGE_ENTITY_NOT_EXACTLY_LINKED';action='Kiểm tra thực thể/alias và ngữ cảnh, không tự gộp acronym chỉ vì giống tên.'
        elif not bridge_common:
            reason='EXPECTED_BRIDGE_EDGES_MISSING';action='Đối chiếu evidence ở A/B và kiểm tra quan hệ bị bỏ sót hoặc gán sai.'
        else:
            reason='EXPECTED_BRIDGE_PATH_PRESENT';action='Có thể thử oracle linking; kiểm tra ý nghĩa quan hệ trước khi truy xuất thực tế.'
        rows.append(dict(id=q['id'],source=q['source'],source_row=q['source_row'],bridge_entity=q['bridge_entity'],
                         A_id=q['hop1_id'],B_id=q['hop2_id'],A_in_KG=a_present,B_in_KG=b_present,
                         bridge_exact_nodes='; '.join(exact),bridge_loose_candidates_NOT_CONFIRMED='; '.join(candidates),
                         A_to_expected_bridge=json.dumps({c:sorted(paper_concepts[a].get(c,set())) for c in exact},ensure_ascii=False),
                         B_to_expected_bridge=json.dumps({c:sorted(paper_concepts[b].get(c,set())) for c in exact},ensure_ascii=False),
                         common_semantic_concepts='; '.join(sorted(common)),expected_bridge_two_edge_path=bool(bridge_common),
                         strict_USE_EVAL_to_PROPOSES_path=bool(strict_paths),semantic_shortest_hops=sem_distance,
                         full_graph_shortest_hops=full_distance,metadata_only_connection=full_distance is not None and sem_distance is None,
                         reviewed_link_type=relation,available_extraction_A=json.dumps(available.get(q['hop1_id'],[])),
                         available_extraction_B=json.dumps(available.get(q['hop2_id'],[])),status=reason,next_action=action))
    targets={pid for q in qa for pid in q['gold_ids']}
    missing_nodes=sorted(pid for pid in targets if nodes.get(f'paper:{pid}',{}).get('type')!='Paper')
    need_extract=[pid for pid in missing_nodes if pid not in available]
    suffix='' if args.tag=='v3' else '_'+args.tag
    diagnostic_path=ROOT/'data/processed'/f'bridge_{args.version}_dev_missing_kg_papers{suffix}.jsonl'
    diagnostic_path.write_text(''.join(json.dumps(corpus[pid],ensure_ascii=False)+'\n' for pid in need_extract),encoding='utf-8')
    out=ROOT/'results'/f'bridge_{args.version}';out.mkdir(parents=True,exist_ok=True)
    with (out/f'kg_audit_dev{suffix}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(split='dev',n_questions=len(qa),graph=str(graph_path.relative_to(ROOT)),graph_sha256=graph_hash,
                 graph_paper_nodes=sum(d.get('type')=='Paper' for d in nodes.values()),
                 unique_gold_papers=len(targets),unique_gold_papers_in_graph=len(targets)-len(missing_nodes),
                 A_present=sum(r['A_in_KG'] for r in rows),B_present=sum(r['B_in_KG'] for r in rows),
                 both_present=sum(r['A_in_KG'] and r['B_in_KG'] for r in rows),
                 exact_bridge_entity_present=sum(bool(r['bridge_exact_nodes']) for r in rows),
                 expected_bridge_path=sum(r['expected_bridge_two_edge_path'] for r in rows),
                 strict_USE_EVAL_to_PROPOSES_path=sum(r['strict_USE_EVAL_to_PROPOSES_path'] for r in rows),
                 any_semantic_path=sum(r['semantic_shortest_hops'] is not None for r in rows),
                 metadata_only_paths=sum(r['metadata_only_connection'] for r in rows),
                 n_missing_papers=len(missing_nodes),n_papers_requiring_new_extraction=len(need_extract),
                 saved_extractions_outside_graph=[pid for pid in missing_nodes if pid in available],
                 link_types=dict(Counter(r['reviewed_link_type'] for r in rows)),status_counts=dict(Counter(r['status'] for r in rows)),
                 diagnostic_extraction_input=str(diagnostic_path.relative_to(ROOT)),
                 diagnostic_warning='These papers were selected from DEV gold endpoints for coverage diagnosis. Extracting only them does not yield a fair full-corpus KG or a valid final Graph-RAG comparison.',
                 path_direction='Semantic audit traverses Paper -> Concept <- Paper using reverse adjacency. Directed A -> Concept -> B paths are not present in this outgoing-edge schema.',
                 test_QA_inspected=False,API_calls=0,KG_modified=False)
    (out/f'kg_audit_dev{suffix}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    assert graph_hash==sha(graph_path),'KG changed during audit'
    assert all(sha(ROOT/'data/eval'/name)==digest for name,digest in manifest['source_workbook_sha256'].items()),'Workbook changed'
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
