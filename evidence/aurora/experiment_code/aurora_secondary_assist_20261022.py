"""One additional six-thread CPU scorer; disjoint arm work via output locks."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/aurora_replication_20261022'
PY='/data/duzc/envs/conda/iclr-gpu-py312/bin/python'


def main():
    lock=(OUT/'secondary_assist.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=json.loads((OUT/'protocol.json').read_text())
    pools=[('calibration',None,None,OUT/'data/calibration')]
    pools += [('test',c,s,OUT/'data/test'/c/str(s)) for c in p['test']['conditions'] for s in p['test']['seeds']]
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false',OMP_WAIT_POLICY='PASSIVE',MALLOC_ARENA_MAX='2')
    pending=pools.copy()
    while pending:
        pipeline=json.loads((OUT/'pipeline_status.json').read_text())['pipeline']
        if pipeline['state']=='failed':
            raise RuntimeError('Main pipeline failed; auxiliary scorer stops')
        progress=False
        for pool in list(pending):
            split,cond,seed,d=pool
            if len(list(d.glob('scores_opt-125m_*.csv')))==10:
                pending.remove(pool);continue
            if not (d/'repairs.npz').exists():continue
            mem={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines() if ':' in r}
            if os.getloadavg()[0]>.6*(os.cpu_count() or 1) or mem['MemAvailable']<16*1024**2:
                continue
            args=['--split',split,'--model','opt-125m','--device','cpu','--threads','6','--reverse-arms']
            if cond:args+=['--condition',cond,'--seed',str(seed)]
            tag='calibration' if cond is None else f'{cond}_{seed}'
            with (OUT/'logs'/f'secondary_assist_{tag}.log').open('a') as log:
                result=subprocess.run(['nice','-n','15',PY,'-u',str(ROOT/'scripts/aurora_score_20261022.py'),*args],
                                      cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError('Auxiliary scorer failed: '+tag)
            assert len(list(d.glob('scores_opt-125m_*.csv')))==10
            pending.remove(pool);progress=True
        if not progress:time.sleep(30)
    print('All secondary pools complete',flush=True)


if __name__=='__main__':main()
