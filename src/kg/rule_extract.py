"""Conservative named-entity context rules; deliberately a weak extraction baseline.

No QA, gold, LLM, entity dictionary from annotations, or corpus graph is read.
ADDRESSES and unnamed concepts are unsupported, and this limitation is measured.
"""
import re
from src.kg.mentions import candidates
from src.kg.aliases import canon,GENERIC

VERSION='local-context-rules-v1'
PROPOSE=re.compile(r'\bwe\s+(?:(?:here|also|further|first)\s+)?(?:propose|introduce|present|develop|release|construct)\b',re.I)
USE=re.compile(r'\b(?:using|uses?|utiliz(?:e|es|ing)|employ(?:s|ing)?|leverag(?:e|es|ing)|based on)\b',re.I)
EVAL=re.compile(r'\b(?:evaluat\w*|experiment\w*|test\w*|benchmark\w*)\b.*\bon\b',re.I)
BASELINE=re.compile(r'\b(?:compar\w*\s+(?:with|to|against)|baselines?\b|outperform\w*\b|surpass\w*\b)',re.I)
NEGATIVE=re.compile(r'\b(?:without|not|never|no)\b',re.I)
DATASET=re.compile(r'\b(?:datasets?|benchmarks?|corpus|corpora|test set)\b',re.I)
SENTENCE=re.compile(r'[^.!?]+(?:[.!?](?!\s*[a-z0-9])|$)',re.I)

def extract_one(paper):
    text=paper['title']+'. '+paper['abstract'];found=[];seen=set()
    # Sentence boundaries preserve numbers and common model tokens reasonably,
    # but are not a scientific parser. Evidence is always the literal source span.
    ranges=list(re.finditer(r'.+?(?:[.!?](?=\s+[A-Z])|$)',text,flags=re.S))
    for sentence in ranges:
        value=sentence.group();mentions=list(candidates(value))
        for raw,key,start,end in mentions:
            if key in GENERIC:continue
            before=value[max(0,start-140):start];local=value[max(0,start-30):min(len(value),end+45)]
            if BASELINE.search(before[-75:]):continue
            rel=None
            verbs=[m for m in USE.finditer(before)]
            last_use=verbs[-1] if verbs else None
            proposes=[m for m in PROPOSE.finditer(before)]
            last_propose=proposes[-1] if proposes else None
            if EVAL.search(before) and DATASET.search(local):rel='EVALUATES_ON'
            elif last_use and (not last_propose or last_use.start()>last_propose.start()):
                if not NEGATIVE.search(before[max(0,last_use.start()-15):]):rel='USES'
            elif last_propose and not NEGATIVE.search(before[last_propose.end():]):rel='PROPOSES'
            if not rel:continue
            label=(rel,canon(raw))
            if label in seen:continue
            seen.add(label)
            found.append(dict(relation=rel,target=raw,target_type='Dataset' if rel=='EVALUATES_ON' or DATASET.search(local) else 'Method',
                              evidence=value,evidence_start=sentence.start(),evidence_end=sentence.end(),evidence_match_kind='literal'))
    return dict(paper_id=paper['paper_id'],extractor=VERSION,relations=found,
                limitations='Named entities only; no ADDRESSES or unnamed concepts. Local context can misattribute roles or types.')
