"""Score the locally frozen margin-pair policies; reuse identical original outputs."""
from pathlib import Path
import argparse,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
from task_readout_score_20261022 import load_metric,calculate,key
from task_readout_runtime_20261022 import resources,dump,sha
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASK=ROOT/'outputs/task_readout_20261022'
OUT=ROOT/'outputs/deep_mechanism_20261022/margin_policies'

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--metric',choices=['nll','bertscore'],required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    resources(a.device,cap=5.5)
    rows={r['index']:r for r in map(json.loads,(TASK/'cohort.jsonl').read_text().splitlines())}
    paths={f:sorted((OUT/'texts'/f).glob('*/[0-9][0-9][0-9][0-9].json')) for f in ['elf','aurora']}
    assert all(len(v)==2048 for v in paths.values()),'Wait for all policy texts before scoring'
    metric=load_metric(a.metric,a.device);stream=torch.cuda.Stream(device=a.device,priority=1)
    stream.wait_stream(torch.cuda.current_stream(a.device));started=time.monotonic();n=0;reused=0
    for family,files in paths.items():
        dest=OUT/'scores'/a.metric/family;dest.mkdir(parents=True,exist_ok=True)
        for i,path in enumerate(files):
            q=json.loads(path.read_text());row=rows[q['index']]
            for arm,text in q['texts'].items():
                k=key(q['index'],text);target=dest/(k+'.json');old=TASK/'scores'/a.metric/family/(k+'.json')
                if target.exists():continue
                if old.exists():
                    prior=json.loads(old.read_text())
                    assert prior['source_sha256']==row['document_sha256'] and prior['reference_sha256']==row['reference_sha256']
                    reused+=1;continue
                with torch.cuda.stream(stream):value,trace=calculate(a.metric,metric,row['document'],row['reference'],text,a.device)
                if trace is not None:
                    np.save(target.with_suffix('.npy'),trace);value['trace_sha256']=sha(target.with_suffix('.npy'))
                dump(target,dict(metric=a.metric,index=q['index'],output_sha256=q['text_sha256'][arm],
                    source_sha256=row['document_sha256'],reference_sha256=row['reference_sha256'],value=value))
                n+=1
            if i%64==63:
                print(a.metric,family,i+1,'new',n,'old_cache_hits',reused,'seconds',round(time.monotonic()-started,1),flush=True)
                time.sleep(.05)
    dump(OUT/f'scores/{a.metric}_complete.json',dict(metric=a.metric,records=4096,new_scores=n,old_cache_hits=reused,
        elapsed=time.monotonic()-started,max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        code_sha256=sha(Path(__file__)),metric_implementation_sha256=sha(ROOT/'scripts/task_readout_score_20261022.py')))

if __name__=='__main__':main()
