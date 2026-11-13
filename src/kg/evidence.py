"""Resolve an evidence quote to a real source span, allowing presentation markup.

No words, numeric values, negation, or unknown TeX commands are discarded.
The returned evidence is always a literal substring of the supplied source.
"""
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class EvidenceSpan:
    text: str
    start: int
    end: int
    match_kind: str

def _normalize(text, formatting=False):
    chars=list(text);offsets=list(range(len(text)))
    if formatting:
        pattern=re.compile(r'\\(?:emph|textit|textbf|textrm|mathrm|mathbf|text)\{([^{}]*)\}')
        for _ in range(8):
            matches=list(pattern.finditer(''.join(chars)))
            if not matches:break
            for m in reversed(matches):
                chars[m.start():m.end()]=chars[m.start(1):m.end(1)]
                offsets[m.start():m.end()]=offsets[m.start(1):m.end(1)]
        text_now=''.join(chars)
        for m in reversed(list(re.finditer(r'\\([%&_])|\\[()\[\]]|\$',text_now))):
            replacement=list(m.group(1) or '')
            original_offset=offsets[m.end()-1]
            chars[m.start():m.end()]=replacement
            offsets[m.start():m.end()]=[original_offset]*len(replacement)
    out=[];mapping=[]
    for c,index in zip(chars,offsets):
        if c.isspace():
            if out and out[-1]!=' ':out.append(' ');mapping.append(index)
        else:
            for lowered in c.lower():out.append(lowered);mapping.append(index)
    if out and out[-1]==' ':out.pop();mapping.pop()
    return ''.join(out),mapping

def locate_evidence(source,quote):
    if not isinstance(quote,str) or not quote.strip():return None
    for formatted in (False,True):
        text,mapping=_normalize(source,formatted)
        needle,_=_normalize(quote,formatted)
        if not needle:continue
        pos=text.find(needle)
        if pos<0:continue
        start=mapping[pos];end=mapping[pos+len(needle)-1]+1
        return EvidenceSpan(source[start:end],start,end,'format_normalized' if formatted else 'literal_normalized')
    return None
