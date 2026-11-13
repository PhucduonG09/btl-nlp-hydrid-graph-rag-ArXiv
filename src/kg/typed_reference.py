"""Corpus-only incidence graph from the independent rule extractor."""
import math
from collections import Counter,defaultdict
from src.kg.aliases import canon,GENERIC
from src.kg.mentions import definitions
from src.retrieval.graph import GraphSearch

def build_typed_graph(papers,extractions,mention_graph,max_df=200):
    by_id={p['paper_id']:p for p in papers};rels=defaultdict(lambda:defaultdict(set));labels={};evidence={}
    for record in extractions:
        pid=record['paper_id'];p=by_id[pid];text=p['title']+'. '+p['abstract'];local=definitions(text)
        for relation in record['relations']:
            assert text[relation['evidence_start']:relation['evidence_end']]==relation['evidence']
            key=canon(relation['target'])
            options=mention_graph['alias_expansions'].get(key,[])
            if key in local:key=local[key]
            elif len(options)==1:key=options[0]
            elif len(options)>1:continue
            if not key or key in GENERIC:continue
            rels[pid][key].add(relation['relation']);labels.setdefault(key,relation['target'])
            evidence.setdefault(pid,{})[key]=dict(start=relation['evidence_start'],end=relation['evidence_end'])
    df=Counter(key for values in rels.values() for key in values);allowed={k for k,n in df.items() if n<=max_df}
    pe={pid:sorted(set(rels[pid])&allowed) for pid in by_id};ep=defaultdict(list)
    for pid,entities in pe.items():
        for entity in entities:ep[entity].append(pid)
    edges={pid:{e:sorted(rels[pid][e]) for e in pe[pid]} for pid in by_id}
    return dict(version='rule-typed-incidence-v1',paper_ids=list(by_id),paper_entities=pe,entity_papers=dict(ep),labels={e:labels[e] for e in allowed},
                title_entities={pid:sorted(set(pe[pid])&set(mention_graph['title_entities'].get(pid,[]))) for pid in by_id},
                alias_expansions=mention_graph['alias_expansions'],edge_relations=edges,evidence=evidence,
                stats=dict(papers=len(by_id),entities=len(ep),incidence_edges=sum(map(len,pe.values())),shared_entities=sum(len(v)>=2 for v in ep.values()),
                           papers_without_relation=sum(not values for values in pe.values())),
                semantics='Local-context rule PROPOSES/USES/EVALUATES_ON only; unvalidated role predictions. Not an LLM KG.')

class TypedGraphSearch(GraphSearch):
    def expand(self,*args,**kwargs):
        ranked,paths=super().expand(*args,**kwargs)
        for path in paths.values():
            for edge in path:
                relations=self.index.data['edge_relations'];entity=edge['entity']
                edge.update(relation='EXTRACTED_RULE_RELATIONS',A_relations=relations[edge['A']][entity],B_relations=relations[edge['B']][entity])
        return ranked,paths

    def strict_onehop(self,seeds,entity_limit=20):
        scores={pid:1/(i+1) for i,pid in enumerate(seeds)};paths={pid:[] for pid in seeds};relations=self.index.data['edge_relations']
        for pid,weight in list(scores.items()):
            entities=sorted((e for e in self.index.paper_entities.get(pid,[]) if len(self.index.entity_papers[e])>=2),key=lambda e:(-self.index.idf(e),e))[:entity_limit]
            for entity in entities:
                if not set(relations[pid][entity])&{'USES','EVALUATES_ON'}:continue
                for other in self.index.entity_papers[entity]:
                    if other==pid or 'PROPOSES' not in relations[other][entity]:continue
                    value=weight*self.index.idf(entity)/(math.log2(len(self.index.entity_papers[entity])+1)*10)*(1.5 if entity in self.index.title_entities.get(other,set()) else 1)
                    scores[other]=scores.get(other,0)+value
                    paths.setdefault(other,[dict(A=pid,entity=entity,B=other,relation='USES_OR_EVAL_TO_PROPOSES',A_relations=relations[pid][entity],B_relations=relations[other][entity])])
        return sorted(scores,key=lambda p:(-scores[p],p)),paths
