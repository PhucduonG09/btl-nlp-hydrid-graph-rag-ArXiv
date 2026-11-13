"""Finite queue-draining helper for the current DEV job; not a scheduled task."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.eval.run_status import read_records
BASE=ROOT/'results/benchmark_v2_assisted/dev_01/generation_prepared'
OUT=BASE/'claude5_controlled'

def main():
    planned=[json.loads(l) for l in (BASE/'jobs_dev.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    expected={(j['id'],j['system']) for j in planned}
    for round_number in range(1,61):
        generated,_=read_records(OUT/'generation.jsonl');judged,_=read_records(OUT/'judge.jsonl')
        g={(r['id'],r['system']):r for r in generated};j={(r['id'],r['system']):r for r in judged}
        available={k for k,r in g.items() if not r.get('error')};complete={k for k,r in j.items() if not r.get('error')}
        print(f'Judge queue round {round_number}: generated={len(available)}/{len(expected)}, judged={len(complete)}/{len(expected)}',flush=True)
        if expected<=complete:return
        if len(g)>=len(expected) and any(g[k].get('error') for k in expected):raise SystemExit('Generation has errors: repair/retry before final judge completion')
        if available-complete:
            command=[sys.executable,str(ROOT/'scripts/run_generation_v2.py'),'--stage','judge-dev','--output-run','claude5_controlled','--available-only','--workers','4']
            if any(r.get('error') for r in j.values()):raise SystemExit('Judge has invalid/error records: inspect before retrying')
            subprocess.run(command,cwd=ROOT,check=True)
        else:time.sleep(10)
    raise SystemExit('Bounded queue wait exhausted; inspect current progress before resuming')
if __name__=='__main__':main()
