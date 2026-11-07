"""Secondary evaluator fixed before inspecting the primary score outcomes."""
import argparse,json,math,time
import numpy as np
import torch
import torch.nn.functional as F
from prospective_tail_20261013 import ROOT,OUT,CONDITIONS,FITS,repairs,writecsv,digest,dump

def freeze():
    dest=OUT/'secondary_protocol.json'
    assert not dest.exists()
    dump(dest,dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),status='secondary addition after initial protocol and generation, before reading any primary scoring outcomes',
        rationale='Check whether equal-agreement repair score effects depend on a single evaluator family.',
        evaluator='local OPT-125M',readout_outputs='Exact same ten frozen text arms as GPT-2-Large; no new selection, regression fitting or repair policy.',
        metric='Same native mask, T5 detokenization, max 1024 evaluator tokens, float16 weights and float32 NLL.',
        inference='Paired random-minus-targeted per-sequence mean NLL and conditional 95% bootstrap; secondary descriptive, not a third confirmatory primary.',
        scope='Sensitivity of external language-model scores, not human preference, factuality or frontier model evaluation.',code_sha256=digest(__import__('pathlib').Path(__file__))))

@torch.inference_mode()
def score(device):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    assert digest(__import__('pathlib').Path(__file__))==json.loads((OUT/'secondary_protocol.json').read_text())['code_sha256']
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.13,device=device)
    tok=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    lt=AutoTokenizer.from_pretrained(ROOT/'models/opt-125m',local_files_only=True);lt.pad_token=lt.eos_token
    lm=AutoModelForCausalLM.from_pretrained(ROOT/'models/opt-125m',local_files_only=True,torch_dtype=torch.float16).to(device).eval()
    for cond in CONDITIONS:
        for seed in json.loads((OUT/'protocol.json').read_text())['generation_seeds']:
            dest=OUT/'test'/cond/str(seed);pack=np.load(dest/'tokens.npz');v=pack['valid'];y=pack['native'];arms={k:pack[k] for k in ['native']+[f'linear_seed{s}' for s in FITS]};arms.update(repairs(pack,cond,seed))
            for method,ids in arms.items():
                fp=dest/f'opt_{method}.csv'
                if fp.exists():continue
                texts=tok.batch_decode(np.where(v,ids,0),skip_special_tokens=True);rows=[];start=time.time()
                for i in range(0,len(texts),4):
                    en=lt(texts[i:i+4],padding=True,truncation=True,max_length=1024,return_tensors='pt').to(device)
                    if en.input_ids.shape[1]<2:sums=[0.]*len(en.input_ids);ns=[0]*len(en.input_ids)
                    else:
                        result=lm(**en);logits=result.logits[:,:-1].float();mask=en.attention_mask[:,1:];loss=F.cross_entropy(logits.transpose(1,2),en.input_ids[:,1:],reduction='none');sums=(loss*mask).sum(1).cpu().tolist();ns=mask.sum(1).cpu().tolist();del result,logits,loss
                    for j,(ss,n) in enumerate(zip(sums,ns)):
                        k=i+j;rows.append(dict(condition=cond,generation_seed=seed,method=method,sequence=k,nll_sum=ss,scored_tokens=n,valid_tokens=int(v[k].sum()),correct=int(((ids[k]==y[k])&v[k]).sum())))
                writecsv(fp,rows);print('OPT scored',cond,seed,method,'PPL',math.exp(sum(r['nll_sum'] for r in rows)/sum(r['scored_tokens'] for r in rows)),'sec',round(time.time()-start,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','score']);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.mode=='freeze':freeze()
    else:score(a.device)
