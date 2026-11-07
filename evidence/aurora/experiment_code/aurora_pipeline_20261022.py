"""Bounded, resumable pipeline. One child per GPU; one CPU secondary scorer.

Jobs write logs to disk. Only task-owned child processes are managed. Test
generation is gated on a hash-verified freeze after calibration scoring.
"""
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/aurora_replication_20261022'
PY='/data/duzc/envs/conda/iclr-gpu-py312/bin/python'
STATUS={};LOCK=threading.Lock()

def record(key,value):
    with LOCK:
        STATUS[key]=value;STATUS['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        tmp=OUT/'pipeline_status.json.tmp';tmp.write_text(json.dumps(STATUS,indent=2)+'\n');tmp.replace(OUT/'pipeline_status.json')

def run(tag,script,args):
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',TORCHINDUCTOR_COMPILE_THREADS='1',
        OMP_WAIT_POLICY='PASSIVE',MALLOC_ARENA_MAX='2')
    command=['nice','-n','10',PY,'-u',str(ROOT/'scripts'/script)]+[str(x) for x in args]
    for attempt in range(1,5):
        with (OUT/'logs'/(tag+'.log')).open('a') as log:
            log.write('\nPIPELINE START '+time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n');log.flush()
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record(tag,dict(state='running',pid=proc.pid,attempt=attempt,command=command,log=str(OUT/'logs'/(tag+'.log'))))
            code=proc.wait()
        if code==0:
            record(tag,dict(state='complete',attempt=attempt));return
        tail=(OUT/'logs'/(tag+'.log')).read_text()[-6000:]
        record(tag,dict(state='failed',exit_code=code,attempt=attempt,tail=tail))
        # Wait only for resource contention; never suppress math/data/weight errors.
        if not any(s in tail for s in ['safe headroom','safe free headroom','Insufficient safe headroom']):
            raise RuntimeError(f'{tag} failed; see its log')
        time.sleep(30)
    raise RuntimeError(f'{tag}: headroom unavailable after bounded retries')

def collect(split,device,condition=None,seed=None,range_start=0,range_end=None):
    args=['collect','--split',split,'--device',device]
    if condition:args+=['--condition',condition,'--seed',seed]
    suffix='_'+condition+'_'+str(seed) if condition else ''
    if range_start or range_end is not None:
        args+=['--range-start',range_start]
        if range_end is not None:args+=['--range-end',range_end]
        suffix+=f'_range{range_start}_{range_end}'
    run('collect_'+split+suffix,'aurora_study_20261022.py',args)

def development_aux():
    collect('training','cuda:0',range_start=1024,range_end=2048)
    collect('calibration','cuda:0')

def development_main():
    collect('training','cuda:2',range_start=0,range_end=1024)
    collect('validation','cuda:2')

def fit_sequence(seeds,device):
    for seed in seeds:run(f'fit_{seed}','aurora_study_20261022.py',['fit','--seed',seed,'--device',device])

def output_and_score(split,device,condition=None,seed=None):
    suffix='_'+condition+'_'+str(seed) if condition else ''
    common=['--split',split]+(['--condition',condition,'--seed',seed] if condition else ['--seed',61022301])
    run('outputs_'+split+suffix,'aurora_study_20261022.py',['outputs',*common,'--device',device])
    run('score_primary_'+split+suffix,'aurora_score_20261022.py',[*common,'--device',device])

def secondary_worker(stop):
    p=json.loads((OUT/'protocol.json').read_text())
    tasks=[('calibration',None,None,OUT/'data/calibration')]
    for cond in p['test']['conditions']:
        for seed in p['test']['seeds']:tasks.append(('test',cond,seed,OUT/'data/test'/cond/str(seed)))
    pending=tasks.copy()
    while pending:
        if stop.is_set():return
        progress=False
        for task in list(pending):
            split,cond,seed,d=task
            if not (d/'repairs.npz').exists():continue
            suffix='_'+cond+'_'+str(seed) if cond else ''
            args=['--split',split,'--model','opt-125m','--device','cpu','--threads','6']
            if cond:args+=['--condition',cond,'--seed',seed]
            run('score_secondary_'+split+suffix,'aurora_score_20261022.py',args)
            pending.remove(task);progress=True
        if not progress:time.sleep(10)

def test_sequence(jobs,device):
    for cond,seed in jobs:
        collect('test',device,cond,seed)
        output_and_score('test',device,cond,seed)

def main():
    lock=(OUT/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record('pipeline',dict(state='starting',pid=os.getpid(),started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    if not (OUT/'protocol.json').exists():raise RuntimeError('Freeze the pilot-reviewed protocol first')
    stop=threading.Event()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(development_main);b=pool.submit(development_aux)
            a.result();b.result()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(fit_sequence,[17,43],'cuda:2');b=pool.submit(fit_sequence,[29],'cuda:0')
            a.result();b.result()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as cpu:
            secondary=cpu.submit(secondary_worker,stop)
            try:
                output_and_score('calibration','cuda:2')
                run('freeze_test','aurora_study_20261022.py',['freeze-test'])
                p=json.loads((OUT/'protocol.json').read_text())
                jobs=[(cond,seed) for cond in p['test']['conditions'] for seed in p['test']['seeds']]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    a=pool.submit(test_sequence,jobs[::2],'cuda:2');b=pool.submit(test_sequence,jobs[1::2],'cuda:0')
                    a.result();b.result()
                secondary.result()
            finally:
                stop.set()
        for model in ['gpt2-large','opt-125m']:
            run('analyze_'+model,'aurora_analyze_20261022.py',['--model',model])
        run('integrity','aurora_verify_20261022.py',[])
        record('pipeline',dict(state='experiments_complete_pending_manuscript_review',finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    except BaseException as e:
        stop.set();record('pipeline',dict(state='failed',error=repr(e)));raise

if __name__=='__main__':main()
