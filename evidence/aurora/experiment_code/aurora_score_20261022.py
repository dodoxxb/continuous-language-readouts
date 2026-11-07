"""Score frozen AURORA output edits; bounded memory, no generation feedback."""
import argparse
import fcntl
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from aurora_study_20261022 import ROOT,OUT,FITS,POLICIES,source_mask,data_dir,csvwrite,protocol,verify_protocol

@torch.inference_mode()
def score(d,model_name,device,threads=4,batch_size=1,reverse_arms=False):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    verify_protocol()
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    if device.startswith('cuda'):
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<5.5*2**30:raise RuntimeError('Scoring needs 5.5 GiB safe headroom')
        torch.cuda.set_per_process_memory_fraction(3.5*2**30/total,device)
    native_tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2',local_files_only=True)
    tok=AutoTokenizer.from_pretrained(ROOT/'models'/model_name,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained(ROOT/'models'/model_name,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    y=np.load(d/'y.npy');v=source_mask(y);repaired=np.load(d/'repairs.npz')
    arms={'native':y}
    arms.update({f'linear{s}':np.load(d/f'linear{s}.npy') for s in FITS})
    arms.update({f'repair_{k}':repaired[k] for k in POLICIES})
    arm_order=list(arms.items())
    if reverse_arms:arm_order.reverse()
    for name,raw in arm_order:
        target=d/f'scores_{model_name}_{name}.csv'
        if target.exists():continue
        # Two bounded CPU workers traverse arms in opposite order. Each holds
        # at most one lock and rechecks the atomic CSV after acquiring it.
        # This changes scheduling only, never a sequence's numerical forward.
        with target.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if target.exists():continue
            ids=np.where(v,raw,y)
            texts=native_tok.batch_decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            texts=[text if y[i,0]==50256 else native_tok.eos_token+text for i,text in enumerate(texts)]
            # No truncation: mark long cases ineligible rather than editing unscored tails.
            encoded=tok(texts,add_special_tokens=True,truncation=False)['input_ids']
            rows=[];started=time.monotonic()
            for i in range(0,len(texts),batch_size):
                batch=encoded[i:i+batch_size]
                for j,seq in enumerate(batch):
                    k=i+j;common=dict(sequence=k,method=name,evaluator=model_name,device=device,
                        dtype='bfloat16 weights/compute; float32 cross entropy',uncapped_tokens=len(seq),
                        valid_tokens=int(v[k].sum()),correct=int(((ids[k]==y[k])&v[k]).sum()))
                    if not 2<=len(seq)<=1024:
                        rows.append(dict(**common,nll_sum=0.,tokens=0,eligible=0));continue
                    inp=torch.tensor([seq],dtype=torch.long,device=device)
                    with torch.autocast(device.split(':')[0],dtype=torch.bfloat16):
                        output=lm(input_ids=inp,use_cache=False)
                    logits=output.logits[:,:-1].float();targets=inp[:,1:]
                    loss=F.cross_entropy(logits.transpose(1,2),targets,reduction='sum')
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite evaluator loss')
                    rows.append(dict(**common,nll_sum=float(loss),tokens=len(seq)-1,eligible=1))
                    del output,logits,loss,inp
                if (i+batch_size)%64==0:
                    print('score',d.name,model_name,name,i+batch_size,'sec',round(time.monotonic()-started,1),flush=True)
                    time.sleep(.05)
            csvwrite(target,rows)
            print('scored',model_name,name,'sequences',len(rows),'sec',round(time.monotonic()-started,1),flush=True)
    del lm;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',default='calibration');p.add_argument('--condition')
    p.add_argument('--seed',type=int);p.add_argument('--model',default='gpt2-large',choices=['gpt2-large','opt-125m'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--reverse-arms',action='store_true')
    a=p.parse_args()
    from contextlib import nullcontext
    ctx=torch.cuda.stream(torch.cuda.Stream(device=a.device,priority=1)) if a.device.startswith('cuda') else nullcontext()
    with ctx:score(data_dir(a.split,a.condition,a.seed),a.model,a.device,a.threads,reverse_arms=a.reverse_arms)

if __name__=='__main__':main()
