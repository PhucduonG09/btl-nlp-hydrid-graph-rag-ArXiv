"""Explicit human answer ratings, kept separate from automated judge output."""
import sqlite3
from datetime import datetime,timedelta,timezone
from pathlib import Path
QUALITY={'correct','partial','incorrect','abstained','uncertain'}
SUPPORT={'all_supported','mixed','unsupported','not_applicable','uncertain'}

def resolved_ratings(rows, allowed_ids):
    """Require explicit agreement; never resolve disagreement by last writer."""
    grouped={sample:[] for sample in allowed_ids}
    for row in rows:
        if row['sample_id'] in grouped:grouped[row['sample_id']].append(row)
    resolved={};pending={}
    for sample,group in grouped.items():
        if not group:pending[sample]='unrated'
        elif any(r['reference_issue'] for r in group):pending[sample]='reference_issue'
        elif any(r['quality']=='uncertain' or r['support']=='uncertain' for r in group):pending[sample]='uncertain'
        elif len({(r['quality'],r['support']) for r in group})!=1:pending[sample]='disagreement'
        else:resolved[sample]=dict(quality=group[0]['quality'],support=group[0]['support'],n_reviewers=len(group))
    return resolved,pending
def resolved_ratings_v2(rows, allowed_ids):
    """Amendment A1 (2026-11-11): reference_issue is a stratum flag, not a blocker.
    A sample is resolved only when every rater gave the same (quality, support) and none is 'uncertain'.
    Disagreement is never resolved by last writer. The flag is kept so reports can stratify."""
    grouped={sample:[] for sample in allowed_ids}
    for row in rows:
        if row['sample_id'] in grouped:grouped[row['sample_id']].append(row)
    resolved={};pending={}
    for sample,group in grouped.items():
        if not group:pending[sample]='unrated'
        elif any(r['quality']=='uncertain' or r['support']=='uncertain' for r in group):pending[sample]='uncertain'
        elif len({(r['quality'],r['support']) for r in group})!=1:pending[sample]='disagreement'
        else:resolved[sample]=dict(quality=group[0]['quality'],support=group[0]['support'],n_reviewers=len(group),
            reference_issue=any(bool(r['reference_issue']) for r in group),reference_issue_all=all(bool(r['reference_issue']) for r in group))
    return resolved,pending
def ratings(path):
    if not Path(path).exists():return []
    with sqlite3.connect(path) as conn:
        conn.row_factory=sqlite3.Row
        exists=conn.execute("SELECT name FROM sqlite_master WHERE name='ratings'").fetchone()
        return [dict(r) for r in conn.execute('SELECT * FROM ratings')] if exists else []
def save_rating(path,*,sample_id,reviewer,quality,support,note='',reference_issue=False):
    if not reviewer.strip():raise ValueError('Điền tên người chấm.')
    if quality not in QUALITY or support not in SUPPORT:raise ValueError('Chọn đủ hai nhãn chấm.')
    if quality=='abstained' and support!='not_applicable':raise ValueError('Đáp án từ chối: chọn hỗ trợ nguồn là Không áp dụng.')
    if quality!='abstained' and support=='not_applicable':raise ValueError('Không áp dụng chỉ dùng cho đáp án từ chối.')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS ratings(sample_id TEXT, reviewer TEXT, quality TEXT, support TEXT, note TEXT, reference_issue INTEGER, reviewed_at TEXT, PRIMARY KEY(sample_id,reviewer))')
        conn.execute('INSERT OR REPLACE INTO ratings VALUES(?,?,?,?,?,?,?)',(sample_id,reviewer.strip(),quality,support,note.strip(),int(reference_issue),datetime.now(timezone(timedelta(hours=7))).isoformat()))
