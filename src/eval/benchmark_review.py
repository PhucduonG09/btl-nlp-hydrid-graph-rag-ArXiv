"""Local review decisions stored separately from immutable benchmark inputs."""
import hashlib,json,sqlite3
from datetime import datetime,timedelta,timezone
from pathlib import Path

def fingerprint(row):
    fields={k:row[k] for k in ['id','type','question','reference_answer','gold_ids']}
    fields['source_text_sha256']=row.get('source_text_sha256')
    return hashlib.sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def decisions(path):
    if not Path(path).exists():return {}
    with sqlite3.connect(path) as conn:
        conn.row_factory=sqlite3.Row
        found=conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='decisions'").fetchone()
        if not found:return {}
        return {r['id']:dict(r) for r in conn.execute('SELECT * FROM decisions')}

def effective(row,stored):
    result=dict(row);decision=stored.get(row['id'])
    if decision and decision['source_sha256']==fingerprint(row):
        result.update(question=decision['question'],reference_answer=decision['answer'],review_status=decision['status'],
                      reviewer=decision['reviewer'],review_note=decision['note'],reviewed_at=decision['updated_at'])
    return result

def content_status(row,audit_rows):
    """Source-check status is separate from independent human approval."""
    if row.get('review_status') in {'approved','approved_existing'}:return 'confirmed'
    if row.get('review_status')=='rejected':return 'needs_replacement'
    audit=audit_rows.get(row['id'])
    if audit and audit.get('source_fingerprint')==fingerprint(row):return audit['status']
    return 'needs_review'

def curated_rows(rows,audit_rows,allow_assisted=False):
    selected=[]
    for row in rows:
        status=content_status(row,audit_rows)
        if status=='confirmed':
            kind='prior_bridge_approval' if row.get('origin')=='bridge_v1_approved' else 'user_confirmation'
        elif status=='cleared' and allow_assisted:kind='assistant_source_review'
        else:continue
        selected.append(dict(row,curation_provenance=kind))
    return selected

def save(path,row,*,question,answer,status,reviewer,note=''):
    if row.get('origin')=='bridge_v1_approved':raise ValueError('Approved bridge v1 is read-only in this review.')
    if status not in {'pending','approved','rejected'}:raise ValueError('Invalid review status')
    if not reviewer.strip():raise ValueError('Điền tên người duyệt trước khi lưu.')
    if not question.strip() or not answer.strip():raise ValueError('Câu hỏi và đáp án không được để trống.')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path,timeout=15) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY,source_sha256 TEXT NOT NULL,question TEXT NOT NULL,answer TEXT NOT NULL,status TEXT NOT NULL,reviewer TEXT NOT NULL,note TEXT NOT NULL,updated_at TEXT NOT NULL)')
        conn.execute('INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)',
                     (row['id'],fingerprint(row),question.strip(),answer.strip(),status,reviewer.strip(),note.strip(),
                      datetime.now(timezone(timedelta(hours=7))).isoformat()))
