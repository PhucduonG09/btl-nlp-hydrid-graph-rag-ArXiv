import json, os, random, subprocess, time, hashlib
import numpy as np, yaml

def load_config(*paths):
    """Load base.yaml rồi ghi đè bằng file thí nghiệm (merge nông theo từng khóa cấp 1)."""
    cfg = {}
    for p in paths:
        with open(p) as f:
            for k, v in (yaml.safe_load(f) or {}).items():
                cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfg

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def log_run(cfg, system_id, metrics, path="results/runs.jsonl"):
    """Mỗi dòng = một lần chạy. experiment_id là hash của config để phát hiện chạy trùng."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cfg_str = json.dumps(cfg, sort_keys=True)
    rec = {
        "experiment_id": hashlib.md5(cfg_str.encode()).hexdigest()[:10],
        "system": system_id, "time": time.strftime("%Y-%m-%d %H:%M"),
        "commit": git_commit(), "config": cfg, "metrics": metrics,
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec["experiment_id"]