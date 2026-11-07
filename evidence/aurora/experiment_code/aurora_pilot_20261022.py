"""Mechanical pilot, never used in the scientific fitting/calibration/test pools."""
import argparse
import json
import time
from contextlib import nullcontext
import numpy as np
import torch
from aurora_runtime_20261022 import OUT,setup,capture,capture_streamed,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--steps',type=int,default=4)
    p.add_argument('--memory-gib',type=float,default=3.5);a=p.parse_args()
    tag=a.device.replace(':','_')+f'_batch{a.batch}_steps{a.steps}'
    dest=OUT/'pilot'/tag;dest.mkdir(parents=True,exist_ok=True)
    g=setup(a.device,a.threads,a.memory_gib)
    ctx=torch.cuda.stream(torch.cuda.Stream(device=a.device,priority=1)) if a.device.startswith('cuda') else nullcontext()
    with ctx:
        public,_,stats=capture(g,1,2,32,61022001,1)
        adapted,_,_=capture_streamed(g,1,2,32,61022001)
        checks={k:bool(np.array_equal(public[k],adapted[k])) for k in public}
        if not all(checks.values()):raise AssertionError(('Public/adapter mismatch',checks))
        dump(dest/'parity.json',dict(public_vs_streamed_exact=checks,stats=stats))
        print('Public/adapter parity passed',checks,flush=True)
        pack,texts,stats=capture_streamed(g,a.batch,a.steps,512,61022002)
        np.savez(dest/'states.npz',**pack)
        dump(dest/'benchmark.json',dict(arguments=vars(a),stats=stats,texts=texts))
        print('Benchmark',json.dumps(stats),flush=True)

if __name__=='__main__':main()
