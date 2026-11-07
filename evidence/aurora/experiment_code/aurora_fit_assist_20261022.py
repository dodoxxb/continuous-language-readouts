"""Use the freed validation GPU for prespecified fits while calibration samples."""
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
    lock=(OUT/'fit_assist.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        status=json.loads((OUT/'pipeline_status.json').read_text())
        if status['pipeline']['state']=='failed':raise RuntimeError('Main pipeline failed')
        if status.get('collect_validation',{}).get('state')=='complete':break
        time.sleep(15)
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false',OMP_WAIT_POLICY='PASSIVE',MALLOC_ARENA_MAX='2')
    for seed in [17,43]:
        if (OUT/f'fits/seed{seed}/selected.json').exists():continue
        command=['nice','-n','10',PY,'-u',str(ROOT/'scripts/aurora_study_20261022.py'),
                 'fit','--seed',str(seed),'--device','cuda:2']
        with (OUT/'logs'/f'fit_assist_{seed}.log').open('a') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            # The main pipeline will still execute the same locked seed task.
            raise RuntimeError('Development fit helper failed; main task remains queued: '+str(seed))
    print('Prespecified GPU2 fits complete',flush=True)


if __name__=='__main__':main()
