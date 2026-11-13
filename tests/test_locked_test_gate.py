import hashlib,json
import pytest
from scripts.run_locked_test_v2 import calibration_gate
from src.eval.human_calibration import save_rating

def pack(root):
    base=root/'data/eval/generation_calibration_v2';base.mkdir(parents=True)
    items=''.join(json.dumps(dict(sample_id=f'H{i:03d}'))+'\n' for i in range(1,51))
    (base/'blind_items.jsonl').write_bytes(items.encode('utf-8'))
    (base/'manifest.json').write_text(json.dumps(dict(items_sha256=hashlib.sha256(items.encode()).hexdigest())))
    return base

def test_missing_human_ratings_stops_before_test_access(tmp_path):
    pack(tmp_path)
    # No test file even exists: the intended human gate, not a file read, fails.
    with pytest.raises(ValueError,match='0/50.*No test QA opened'):calibration_gate(tmp_path)

def test_disagreement_blocks_gate_even_if_all_samples_have_a_rating(tmp_path):
    base=pack(tmp_path)
    for i in range(1,51):save_rating(base/'ratings.sqlite3',sample_id=f'H{i:03d}',reviewer='fixture-A',quality='correct',support='all_supported')
    assert calibration_gate(tmp_path)['n_resolved']==50
    save_rating(base/'ratings.sqlite3',sample_id='H001',reviewer='fixture-B',quality='partial',support='all_supported')
    with pytest.raises(ValueError,match='49/50'):calibration_gate(tmp_path)
