"""Frozen prospective readout-quality test and equal-error repair intervention.

The native labels and margins are diagnostic oracles, not an inference policy.
Calibration reuses previously inspected data; all test generations are new.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/confirmatory_readout_20261013'
OUT=ROOT/'outputs/prospective_tail_20261013'
FITS=[17,29,43]
CONDITIONS={
    'B32':dict(model='ELF-B-owt-torch',checkpoint='checkpoint_95085',steps=32,gamma=1.5,cfg=3.),
    'M32':dict(model='ELF-M-owt-torch',checkpoint='checkpoint_76070',steps=32,gamma=1.5,cfg=3.),
    'PD8':dict(model='ELF-B-PD-owt-torch',checkpoint='checkpoint_19017',steps=8,gamma=1.5,cfg=2.5),
}
BASE=['error_rate','log_length','native_nll']
TAIL=BASE+['rare_error_rate','numeric_error_rate','lowmargin_error_rate']
POLICIES=['repair_random11','repair_random23','repair_random37','repair_rare','repair_numeric','repair_lowmargin']

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def writecsv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def readcsv(path):return list(csv.DictReader(path.open()))

def groups():
    return np.load(OUT/'token_counts.npy'),np.load(OUT/'numeric.npy')

def features(y,v,pred,margin,native_nll):
    counts,numeric=groups();e=(pred!=y)&v;d=v.sum(1).clip(1)
    return np.stack([e.sum(1)/d,np.log(d),native_nll,
        (e&(counts[y]<100)).sum(1)/d,(e&numeric[y]).sum(1)/d,
        (e&(margin<2)).sum(1)/d],axis=1)

def fit_ridge(x,y):
    mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-12,sd,1.)
    z=np.c_[np.ones(len(x)),(x-mu)/sd];reg=np.eye(z.shape[1])*10.;reg[0,0]=0
    beta=np.linalg.solve(z.T@z+reg,z.T@y)
    return dict(mean=mu.tolist(),std=sd.tolist(),beta=beta.tolist())

def predict(x,fit):
    return np.c_[np.ones(len(x)),(x-np.array(fit['mean']))/np.array(fit['std'])]@np.array(fit['beta'])

def freeze():
    if (OUT/'frozen.json').exists():
        verify_frozen();print('Already frozen',flush=True);return
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'test').exists():raise RuntimeError('Cannot freeze after creating tests')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    counts=np.bincount(np.load(OLD/'data/elf/train/y.npy',mmap_mode='r'),minlength=32128)
    numeric=np.array([any(c.isdigit() for c in tok.convert_ids_to_tokens(i)) for i in range(32100)]+[False]*28)
    np.save(OUT/'token_counts.npy',counts);np.save(OUT/'numeric.npy',numeric)
    protocol=dict(date='2026-10-13',hypothesis='Tail-specific native-disagreement burdens improve out-of-condition prediction of excess evaluator NLL beyond aggregate disagreement, length and native evaluator NLL.',
        calibration='Previously inspected 512 ELF-B SDE32 samples, seed 61013001; exploratory calibration, not an untouched development set.',
        readouts='All three already validation-selected 16384-sequence ELF-B affine maps; frozen transfer, no target-domain fitting.',
        fit_seeds=FITS,conditions=CONDITIONS,generation_seeds=[61013201,61013202],sequences_per_seed=256,batch_size=4,
        regression='Separate ridge per readout fit, training-only standardization; alpha=10, unpenalized intercept; no hyperparameter search.',
        baseline_features=BASE,tail_features=TAIL,rare='native target training count <100 including unseen, from the fixed 16384-sequence training pool',
        numeric='native tokenizer token contains a Unicode digit',lowmargin='native top1-top2 margin <2',
        primary='Reduction in mean absolute error for per-sequence (linear NLL/token minus native NLL/token), averaged over three fixed readouts and equally weighted over conditions/seeds.',
        secondary=['Per-condition MAE, RMSE, R2 and Spearman','Native agreement and autonomous-EOS agreement','Exact per-sequence matched-error repair'],
        intervention='For prespecified fit seed 17, correct floor(number_of_native_disagreements/2) errors in each sequence using native labels. Select errors by ascending training count, numeric-first, ascending native margin, or three uniform random permutations. The error count is identical across repair arms. Oracle diagnostic, no claim of usable routing or speedup.',
        repair_ties='Random tie-breaks fixed per condition, generation seed, sequence and policy; no outcome-based choice.',
        repair_primary='Equal-weight mean of paired per-sequence NLL/token difference: average random repair minus rare-first repair. Also report numeric and low-margin controls, corpus PPL and evaluator token counts.',
        inference='4000 sequence-cluster paired bootstrap draws, stratified by condition and generation seed; average fitted-readout errors inside each sequence. Conditional on checkpoints/readouts, no denoiser-training-seed inference. Bonferroni 97.5% intervals for the two primary directional tests; 95% descriptive intervals otherwise.',
        evaluation='GPT-2-Large, float16 weights, float32 losses, native valid mask shared before T5 detokenization, 1024 evaluator-token cap. Diagnostic conditional text score, not factuality or autonomous generation.',
        failure_policy='Retain every condition, fit and sign. No retuning against these tests. Failed hypotheses remain reported.',
        generalization_scope='B is a fresh-seed replication; M and released PD change checkpoint/training/scale jointly, not independent randomized training replicas or cross-architecture tests.')
    dump(OUT/'protocol.json',protocol)
    dat=OLD/'data/elf/fresh_test';y=np.load(dat/'y.npy');v=np.load(dat/'valid.npy');m=np.load(dat/'native_margin.npy')
    rows=readcsv(OLD/'ppl/per_sequence.csv');scores={}
    for method in ['native']+[f'linear_seed{s}' for s in FITS]:
        rr=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['sequence_index']))
        assert len(rr)==len(y) and all(int(r['scored_tokens'])>0 for r in rr)
        scores[method]=np.array([float(r['nll_sum'])/int(r['scored_tokens']) for r in rr])
    models={};cal=[]
    for seed in FITS:
        pred=np.load(OLD/f'fits/elf/n16384_seed{seed}/fresh_test.npz')['pred']
        x=features(y,v,pred,m,scores['native']);target=scores[f'linear_seed{seed}']-scores['native']
        models[str(seed)]={'baseline':fit_ridge(x[:,:3],target),'tail':fit_ridge(x,target)}
        for i in range(len(x)):
            cal.append(dict(fit_seed=seed,sequence=i,target=float(target[i]),**{k:float(x[i,j]) for j,k in enumerate(TAIL)}))
    dump(OUT/'predictors.json',models);writecsv(OUT/'calibration.csv',cal)
    paths=[OUT/'protocol.json',OUT/'predictors.json',OUT/'calibration.csv',OUT/'token_counts.npy',OUT/'numeric.npy',Path(__file__)]
    paths += [OLD/f'fits/elf/n16384_seed{s}/selected.pt' for s in FITS]
    paths += [ROOT/'models'/c['model']/c['checkpoint'] for c in CONDITIONS.values()]
    dump(OUT/'frozen.json',dict(frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),files={str(p.relative_to(ROOT)):digest(p) for p in paths},test_generated=False))
    print('Frozen protocol and fitted predictors before test generation',digest(OUT/'protocol.json'),flush=True)

def verify_frozen():
    f=json.loads((OUT/'frozen.json').read_text())
    for p,h in f['files'].items():
        assert digest(ROOT/p)==h,f'Frozen input changed: {p}'

@torch.inference_mode()
def generate(condition,device):
    sys.path.insert(0,str(ROOT/'outputs/phase0_phase1_20260520/scripts'))
    from phase2_trajectory_audit import load_elf,forward_x,decode_logits,get_sampling_steps,mask_after_eos
    p=json.loads((OUT/'protocol.json').read_text());cnd=CONDITIONS[condition];md=ROOT/'models'/cnd['model']
    args=SimpleNamespace(root=ROOT,out=OUT/'test'/condition,encoder_model=ROOT/'models/t5-small',tokenizer=ROOT/'models/t5-small',ppl_model=ROOT/'models/gpt2-large',config=md/'config.yml',checkpoint=md/cnd['checkpoint'],weights='ema_params1',num_samples=p['sequences_per_seed'],batch_size=p['batch_size'])
    cwd=Path.cwd();os.chdir(ROOT/'official-ELF-pytorch_elf')
    try:c,tok,model=load_elf(args,torch.device(device))
    finally:os.chdir(cwd)
    readouts={}
    for seed in FITS:
        state=torch.load(OLD/f'fits/elf/n16384_seed{seed}/selected.pt',weights_only=True,map_location=device)
        linear=torch.nn.Linear(state['weight'].shape[1],state['weight'].shape[0]).to(device);linear.load_state_dict(state);readouts[seed]=linear.eval()
    for seed in p['generation_seeds']:
        dest=OUT/'test'/condition/str(seed);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'generation.json').exists():continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);gen=torch.Generator().manual_seed(seed)
        xs=[];ys=[];ms=[];preds={s:[] for s in FITS};gaps={s:[] for s in FITS};started=time.time()
        for start in range(0,args.num_samples,args.batch_size):
            b=min(args.batch_size,args.num_samples-start)
            steps=get_sampling_steps(cnd['steps'],'logit_normal',c.denoiser_p_mean,c.denoiser_p_std,torch.device(device),torch.float32,gen)
            z=torch.randn(b,c.max_length,model.text_encoder_dim,device=device)*c.denoiser_noise_scale;sc=torch.zeros_like(z)
            for k in range(cnd['steps']):
                t=float(steps[k]);tn=float(steps[k+1]);tm=t;zm=z
                if k<cnd['steps']-1:
                    alpha=max(0.,min(1.,1.-cnd['gamma']*(tn-t)));tm=alpha*t
                    zm=alpha*z+(1.-alpha)*torch.randn_like(z)*c.denoiser_noise_scale
                xp,_=forward_x(model,zm,tm,sc,cnd['cfg'],c,False)
                z=zm+(tn-tm)*(xp-zm)/max(1.-tm,float(c.t_eps));sc=xp
            xs.append(z.float().cpu())
            for j in range(b):
                logits=decode_logits(model,z[j:j+1],1.,cnd['cfg'],c).float();top=logits.topk(2,-1)
                ys.append(mask_after_eos(top.indices[...,0],1,0).cpu());ms.append((top.values[...,0]-top.values[...,1]).cpu())
                del logits,top
                for s,linear in readouts.items():
                    top=linear(z[j].float()).topk(2,-1)
                    preds[s].append(top.indices[:,0].cpu().numpy());gaps[s].append((top.values[:,0]-top.values[:,1]).cpu().numpy())
            if start%64==0:print(condition,seed,'generated',start+b,'seconds',round(time.time()-started,1),flush=True)
        x=torch.cat(xs).numpy();y=torch.cat(ys).numpy();m=torch.cat(ms).numpy()
        np.save(dest/'x.npy',x);np.savez_compressed(dest/'tokens.npz',native=y,valid=y!=0,margin=m,**{f'linear_seed{s}':np.stack(preds[s]) for s in FITS},**{f'gap_seed{s}':np.stack(gaps[s]) for s in FITS})
        dump(dest/'generation.json',dict(condition=condition,seed=seed,samples=len(y),elapsed_seconds=time.time()-started,protocol_sha256=digest(OUT/'protocol.json'),tokens_sha256=digest(dest/'tokens.npz'),x_sha256=digest(dest/'x.npy')))
    del model,readouts;gc.collect();torch.cuda.empty_cache()

def repairs(pack,condition,seed):
    y=pack['native'];v=pack['valid'];pred=pack['linear_seed17'];counts,numeric=groups()
    result={k:pred.copy() for k in POLICIES};ci=list(CONDITIONS).index(condition)
    for i in range(len(y)):
        errors=np.flatnonzero((pred[i]!=y[i])&v[i]);n=len(errors)//2
        for pi,policy in enumerate(POLICIES):
            rng=np.random.default_rng(np.random.SeedSequence([seed,ci,i,pi,61013]));tie=rng.random(len(errors))
            if policy.startswith('repair_random'):chosen=rng.permutation(errors)[:n]
            else:
                score=counts[y[i,errors]] if policy=='repair_rare' else -numeric[y[i,errors]].astype(int) if policy=='repair_numeric' else pack['margin'][i,errors]
                chosen=errors[np.lexsort((tie,score))[:n]]
            result[policy][i,chosen]=y[i,chosen]
            assert int(((result[policy][i]!=y[i])&v[i]).sum())==len(errors)-n
    return result

@torch.inference_mode()
def score(device):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(ROOT/'models/t5-small',local_files_only=True)
    lm_tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2-large',local_files_only=True);lm_tok.pad_token=lm_tok.eos_token
    lm=AutoModelForCausalLM.from_pretrained(ROOT/'models/gpt2-large',local_files_only=True,torch_dtype=torch.float16).to(device).eval()
    protocol=json.loads((OUT/'protocol.json').read_text())
    for cond in CONDITIONS:
        for seed in protocol['generation_seeds']:
            dest=OUT/'test'/cond/str(seed);pack=np.load(dest/'tokens.npz');v=pack['valid'];y=pack['native']
            arms={k:pack[k] for k in ['native']+[f'linear_seed{s}' for s in FITS]};arms.update(repairs(pack,cond,seed))
            # Persist intervention outputs so exact agreement equality can be independently checked.
            np.savez_compressed(dest/'repairs.npz',**{k:arms[k] for k in POLICIES})
            for method,ids in arms.items():
                fp=dest/f'ppl_{method}.csv'
                if fp.exists():continue
                texts=tok.batch_decode(np.where(v,ids,0),skip_special_tokens=True);rows=[];start=time.time()
                for i in range(0,len(texts),2):
                    en=lm_tok(texts[i:i+2],padding=True,truncation=True,max_length=1024,return_tensors='pt').to(device)
                    if en.input_ids.shape[1]<2:
                        sums=[0.]*len(en.input_ids);ns=[0]*len(en.input_ids)
                    else:
                        output=lm(**en);logits=output.logits[:,:-1].float();target=en.input_ids[:,1:];mask=en.attention_mask[:,1:]
                        loss=F.cross_entropy(logits.transpose(1,2),target,reduction='none')
                        sums=(loss*mask).sum(1).cpu().tolist();ns=mask.sum(1).cpu().tolist();del output,logits,loss
                    for j,(ss,n) in enumerate(zip(sums,ns)):
                        k=i+j;valid=v[k];correct=int(((ids[k]==y[k])&valid).sum())
                        rows.append(dict(condition=cond,generation_seed=seed,method=method,sequence=k,nll_sum=ss,scored_tokens=n,valid_tokens=int(valid.sum()),correct=correct))
                writecsv(fp,rows)
                print('scored',cond,seed,method,'PPL',math.exp(sum(r['nll_sum'] for r in rows)/sum(r['scored_tokens'] for r in rows)),'sec',round(time.time()-start,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','generate','score']);ap.add_argument('--condition',choices=list(CONDITIONS));ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    if a.mode=='freeze':freeze();return
    verify_frozen();torch.cuda.set_per_process_memory_fraction(.22,device=a.device)
    stream=torch.cuda.Stream(device=a.device,priority=1)
    with torch.cuda.stream(stream):
        if a.mode=='generate':
            for cond in ([a.condition] if a.condition else CONDITIONS):generate(cond,a.device)
        else:score(a.device)

if __name__=='__main__':main()
