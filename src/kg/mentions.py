"""Corpus-only entity-mention graph, with explicit provenance and no QA inputs.

MENTIONS is not PROPOSES or USES. This graph is an independent retrieval control
and complements, rather than masquerades as, the LLM-extracted semantic KG.
"""
import hashlib
import json
import math
import re
from collections import Counter,defaultdict
from pathlib import Path

from src.kg.aliases import canon,is_acronym,GENERIC

VERSION='mentions-v1'
TOKEN=re.compile(r'(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*(?:\+\+)?(?![A-Za-z0-9])')
DEFINITION=re.compile(r'\(([A-Za-z][A-Za-z0-9+-]{1,15})\)')
STOP=GENERIC|{'llm','llms','mllm','mllms','vlm','vlms','gpu','gpus','cpu','cpus','nlp','sota','url','http','https','github','arxiv','api','apis','pdf','figure','table','abstract','introduction','conclusion','auc','roc','f1','r2','usa','usd'}

def candidates(text):
    for m in TOKEN.finditer(text):
        raw=m.group()
        if raw.lower().endswith('-based'):raw=raw[:-6]
        if len(raw)<3:continue
        if sum(c.isupper() for c in raw)<2 and not (raw[0].isupper() and any(c.isdigit() for c in raw)):continue
        key=canon(raw)
        if key and key not in STOP:yield raw,key,m.start(),m.start()+len(raw)

def definitions(text):
    found={}
    for m in DEFINITION.finditer(text):
        short=m.group(1)
        preceding=re.search(r'([A-Za-z -]+)$',text[max(0,m.start()-200):m.start()].rstrip())
        if not preceding:continue
        words=preceding.group(1).split()[-9:]
        for length in range(len(words),1,-1):
            phrase=' '.join(words[-length:])
            if is_acronym(short,phrase):
                found[canon(short)]=canon(phrase);break
    return found

def build_mention_graph(papers,max_df=200):
    local={};global_expansions=defaultdict(set)
    for p in papers:
        d=definitions(p['title']+'. '+p['abstract']);local[p['paper_id']]=d
        for short,long in d.items():global_expansions[short].add(long)
    entities={};df=Counter();labels={};evidence={};titles={};ambiguous=Counter()
    for p in papers:
        pid=p['paper_id'];text=p['title']+'. '+p['abstract'];names={};title_names=set()
        for raw,key,start,end in candidates(text):
            if key in local[pid]:key=local[pid][key]
            elif len(global_expansions.get(key,set()))==1:key=next(iter(global_expansions[key]))
            elif len(global_expansions.get(key,set()))>1:
                ambiguous[key]+=1;continue
            if key in STOP:continue
            names.setdefault(key,(raw,start,end))
            if start<len(p['title']):title_names.add(key)
        entities[pid]=names;titles[pid]=title_names;df.update(names.keys())
    allowed={key for key,n in df.items() if n<=max_df}
    paper_entities={};entity_papers=defaultdict(list);title_entities={}
    for pid,names in entities.items():
        selected=sorted(set(names)&allowed);paper_entities[pid]=selected
        title_entities[pid]=sorted(titles[pid]&allowed)
        for key in selected:
            raw,start,end=names[key];labels.setdefault(key,raw);entity_papers[key].append(pid)
            evidence.setdefault(pid,{})[key]=dict(start=start,end=end)
    return dict(version=VERSION,paper_ids=[p['paper_id'] for p in papers],paper_entities=paper_entities,
                entity_papers=dict(entity_papers),title_entities=title_entities,labels=labels,evidence=evidence,
                ambiguous_abbreviations_skipped=dict(ambiguous),alias_expansions={k:sorted(v) for k,v in global_expansions.items()},
                stats=dict(papers=len(papers),entities=len(entity_papers),mention_edges=sum(map(len,paper_entities.values())),
                           shared_entities=sum(len(v)>=2 for v in entity_papers.values()),max_df=max_df),
                semantics='Paper MENTIONS Entity, grounded in title/abstract. No assertion of novelty or use. No QA/gold inputs.')

class MentionIndex:
    def __init__(self,data):
        self.data=data;self.paper_entities=data['paper_entities'];self.entity_papers=data['entity_papers']
        self.title_entities={p:set(v) for p,v in data['title_entities'].items()};self.n=len(data['paper_ids'])
    def query_entities(self,query):
        result=[]
        expansions=self.data['alias_expansions']
        for raw,key,_,_ in candidates(query):
            if len(expansions.get(key,[]))==1:key=expansions[key][0]
            elif len(expansions.get(key,[]))>1:continue
            if key in self.entity_papers and key not in result:result.append(key)
        return result
    def idf(self,entity):return math.log1p(self.n/max(1,len(self.entity_papers[entity])))
    def linked_seeds(self,query,fallback,count=3):
        score=defaultdict(float)
        for entity in self.query_entities(query):
            papers=self.entity_papers[entity]
            if len(papers)>10:continue
            for pid in papers:score[pid]+=self.idf(entity)*(2 if entity in self.title_entities.get(pid,set()) else 1)
        linked=[pid for pid,_ in sorted(score.items(),key=lambda x:(-x[1],x[0]))]
        return list(dict.fromkeys(linked+list(fallback)))[:count]
