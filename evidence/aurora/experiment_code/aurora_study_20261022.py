"""Independent-architecture residual prediction and equal-count oracle repair.

Stages are gated: protocol -> development states -> readouts/calibration ->
test freeze -> fresh test states -> scoring -> analysis. No outcome selection.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path
from aurora_runtime_20261022 import ROOT, OUT, UPSTREAM, dump, digest, setup, capture_streamed
import numpy as np
import torch
import torch.nn.functional as F

FITS=[17,29,43]
LENGTH=512
POLICIES=['random11','random23','random37','rare','numeric','lowmargin']
FEATURES=['error_rate','log_length','native_nll','rare_error_rate','numeric_error_rate','lowmargin_error_rate']

def protocol():return json.loads((OUT/'protocol.json').read_text())
def csvwrite(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(path)
def csvread(path):return list(csv.DictReader(path.open()))

def freeze_protocol(batch=8):
    if (OUT/'protocol.json').exists():raise RuntimeError('Protocol exists; cannot silently replace it')
    if (OUT/'data').exists():raise RuntimeError('Development data already exist')
    p=dict(date='2026-10-22',model='AURORA-LM-S OWT; published denoiser EMA and shared text AE',
        hypothesis='Does error composition predict excess evaluator NLL beyond overall error rate, and does equal-count targeted repair improve NLL, on an independent continuous-language architecture?',
        scope='Local affine readout fits in AURORA coordinates. Cross-architecture replication of a protocol; no ELF readout weight transfer. Sampling-condition transfer within one AURORA checkpoint is not checkpoint or training-seed transfer.',
        representation='Raw, denormalized sampler endpoint immediately BEFORE the full six-layer native AE decoder; no decoder hidden-state leakage.',
        sampler=dict(solver='sde_dpmpp',self_cond_cfg_scale=4.0,block_size=16,length=LENGTH,native_early_stop=False),
        training=dict(sequences=2048,steps=64,seed=61022101),validation=dict(sequences=256,steps=64,seed=61022201),
        calibration=dict(sequences=512,steps=64,seed=61022301),
        test=dict(conditions={'S16':16,'S32':32,'S64':64},seeds=[61022401,61022402],sequences_per_seed=256),
        batch_size=batch,fit_seeds=FITS,fit=dict(updates=4096,batch_tokens=1024,learning_rate=0.0005,weight_decay=0.01,noise_aug=0.02,
        validation_updates=[1024,2048,4096],selection='Maximum validation lexical native agreement; earlier update breaks ties; test never used.'),
        boundary='Generate 512-token prefixes with the released 16-token block sampler (the training/default maximum is 1024). OWT is packed text: retain native EOS separators, not first-EOS truncation. Readout fitting and lexical-error analysis exclude native EOS positions. All scored arms share native EOS separators; all-position raw agreement is also reported.',
        score='Decode each edited sequence with native EOS fixed, prepend one EOS literal based on the common native first-token boundary, then re-tokenize with evaluator. GPT-2-Large primary; OPT-125M secondary. float32 losses; BF16 inference; CPU and GPU precision recorded. Common boundary, not autonomous stopping or factual correctness.',
        scoring_note='Prespecified common scorable set: retain a sequence only if every compared arm has 2..1024 evaluator tokens, so no arm is truncated. Persist all outputs and exclusion reasons. The 512-source-token horizon gives room for retokenization. Report the primary and secondary common sets separately.',
        frequency='Counts from native non-EOS training labels only; rare=count<100, including unseen; numeric=Unicode digit in decoded single token; lowmargin=native top1-top2 logit gap<2.',
        frequency_sensitivity=dict(elf_reference_tokens=16754550,reference_count=100,
            definition='Secondary predictor substitutes rare=count/N_AURORA < 100/16754550, matching the ELF relative-frequency cutoff. It is fit on calibration before test. Repair order is unchanged by this positive rescaling. Main count<100 result is always retained.',
            caveat='Matching pools differ in size, tokenizer and generation distribution; this is a within-architecture protocol replication, not an architecture-isolated ranking.'),
        regression=dict(features=FEATURES,baseline_features=FEATURES[:3],alpha=10.0,standardization='calibration only; unpenalized intercept',calibration='512 independent native generations; fit predictor separately for each frozen readout seed'),
        repair='Seed 17 readout; exactly floor(e/2) native corrections within each sequence. Policies: ascending training count, numeric first, ascending native margin, three uniform random permutations. Fixed policy-specific random tie breaks. All final mismatch counts must be identical.',
        inference='4000 paired sequence-bootstrap draws, stratified by condition and generation seed. Shared resampling indices across conditions with common random seeds. Average three readout errors within each sequence. Two primary pooled contrasts use Bonferroni 97.5% CIs; per-condition and other policies use descriptive 95% CIs.',
        primary=['Baseline MAE minus composition-augmented MAE on fresh test sequences','Mean random-repair NLL minus rare-first-repair NLL, using readout seed 17'],
        controls=['All three readout fits','All six repair policies','Frequency-coverage agreement breakdown','Native, linear and repaired evaluator scores','Native EOS and truncation diagnostics','Secondary OPT-125M on exactly the same output edits'],
        failure_policy='Retain every fit, condition, sign, exclusion and failed transfer. No test-based retuning or selection of successful architectures. Oracle native labels/margins diagnose text-score effects; not deployment gates or truth labels.',
        resources='Low-priority GPU stream; sequential native decoding; bounded per-process CUDA allocator; at most two GPU workers on distinct GPUs and two CPU workers (<=8 threads each); no data loader workers or concurrent compilation; logs to files.',
        pilot_disclosure='Pilot only checks downloads, native interface parity, finite outputs, batching/memory/runtime. Pilot sequences never enter fitting, calibration or test.',
        source=json.loads((OUT/'source/upstream_commit.json').read_text()),weights=json.loads((OUT/'source/download_manifest.json').read_text()),
        upstream_files={str(f.relative_to(ROOT)):digest(f) for f in sorted(UPSTREAM.rglob('*')) if f.is_file() and f.suffix in ['.py','.yaml']},
        local_patch='CPU device plumbing only; source/device_utils.upstream.py and source/cpu_device_support.patch preserve the exact change; CUDA sampler/decoder math unchanged.')
    dump(OUT/'protocol.json',p)
    dump(OUT/'protocol_frozen.json',dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),sha256=digest(OUT/'protocol.json')))
    print('Protocol frozen',flush=True)

def verify_protocol():
    assert digest(OUT/'protocol.json')==json.loads((OUT/'protocol_frozen.json').read_text())['sha256']
    for path,sha in protocol()['upstream_files'].items():assert digest(ROOT/path)==sha,path

def data_dir(split,condition=None,seed=None):
    if split=='test':return OUT/'data/test'/condition/str(seed)
    return OUT/'data'/split

def collect(split,device,condition=None,seed=None,memory_gib=3.5,range_start=0,range_end=None):
    verify_protocol();p=protocol()
    if split=='test':
        verify_test_freeze();n=p['test']['sequences_per_seed'];steps=p['test']['conditions'][condition]
    else:
        c=p[split];n=c['sequences'];steps=c['steps'];seed=c['seed']
    d=data_dir(split,condition,seed);d.mkdir(parents=True,exist_ok=True)
    if (d/'ready.json').exists():return
    g=setup(device,threads=2,memory_gib=memory_gib)
    stream=torch.cuda.Stream(device=device,priority=1)
    batch=p['batch_size'];started=time.monotonic()
    assert range_start%batch==0
    range_end=n if range_end is None else min(range_end,n)
    assert range_end==n or range_end%batch==0
    for start in range(range_start,range_end,batch):
        dest=d/f'shard{start:05d}.npz';meta=d/f'shard{start:05d}.json'
        if meta.exists():
            assert digest(dest)==json.loads(meta.read_text())['sha256'];continue
        while torch.cuda.mem_get_info(device)[0]<1.5*2**30:
            torch.cuda.empty_cache();print('Waiting for shared GPU headroom',device,flush=True);time.sleep(10)
        # Common seeds across sampling conditions; independent development splits.
        s=int(np.random.SeedSequence([seed,start,20261022]).generate_state(1)[0])
        with torch.cuda.stream(stream):
            pack,_,stats=capture_streamed(g,min(batch,n-start),steps,LENGTH,s)
        tmp=dest.with_name(dest.name+'.tmp')
        with tmp.open('wb') as f:np.savez(f,**pack)
        tmp.replace(dest)
        dump(meta,dict(start=start,count=len(pack['native']),rng_seed=s,stats=stats,sha256=digest(dest),
            completed_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),protocol_sha256=digest(OUT/'protocol.json')))
        print(split,condition,seed,'completed',start+len(pack['native']),'of',n,'seconds',round(time.monotonic()-started,1),flush=True)
        del pack
        # Yield to shared-server jobs between batches; no busy spin.
        time.sleep(0.5)
    del g;gc.collect();torch.cuda.empty_cache()
    import fcntl
    with (d/'assemble.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (d/'ready.json').exists() and all((d/f'shard{i:05d}.json').exists() for i in range(0,n,batch)):
            assemble(d,n)

def assemble(d,n):
    x=np.lib.format.open_memmap(d/'x.npy',mode='w+',dtype=np.float32,shape=(n,LENGTH,768))
    y=np.empty((n,LENGTH),dtype=np.int32);margin=np.empty((n,LENGTH),dtype=np.float32)
    for f in sorted(d.glob('shard*.npz')):
        start=int(f.stem[5:]);z=np.load(f);end=start+len(z['native'])
        x[start:end]=z['x'];y[start:end]=z['native'];margin[start:end]=z['margin']
    x.flush();del x
    np.save(d/'y.npy',y);np.save(d/'margin.npy',margin)
    dump(d/'ready.json',dict(samples=n,files={k:digest(d/k) for k in ['x.npy','y.npy','margin.npy']},protocol_sha256=digest(OUT/'protocol.json')))

def load_readout(seed,device):
    model=torch.nn.Linear(768,50257).to(device)
    model.load_state_dict(torch.load(OUT/f'fits/seed{seed}/selected.pt',weights_only=True,map_location=device))
    return model.eval()

@torch.inference_mode()
def readout_predictions(model,d,device):
    x=np.load(d/'x.npy',mmap_mode='r');pred=np.empty(x.shape[:2],np.int32)
    for i in range(len(x)):
        for j in range(0,x.shape[1],256):
            xx=torch.from_numpy(np.array(x[i,j:j+256],copy=True)).to(device)
            pred[i,j:j+256]=model(xx.float()).argmax(-1).cpu().numpy()
    return pred

def fit(seed,device):
    # A development-only helper can use the validation GPU while the other GPU
    # finishes calibration sampling. Exactly one writer fits a given seed.
    import fcntl
    dest=OUT/f'fits/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    with (dest/'fit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _fit_unlocked(seed,device)

def _fit_unlocked(seed,device):
    verify_protocol();p=protocol();cfg=p['fit'];d=OUT/'data/training';v=OUT/'data/validation'
    assert (d/'ready.json').exists() and (v/'ready.json').exists()
    dest=OUT/f'fits/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    if device.startswith('cuda'):
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<5*2**30:raise RuntimeError('Fit needs 5 GiB safe free headroom')
        torch.cuda.set_per_process_memory_fraction(2.8*2**30/total,device)
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x=np.load(d/'x.npy',mmap_mode='r').reshape(-1,768);y=np.load(d/'y.npy').reshape(-1)
    ix=np.flatnonzero(y!=50256);yv=np.load(v/'y.npy');valid=yv!=50256
    model=torch.nn.Linear(768,50257).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    started=time.monotonic();best=-1.;history=[]
    for step in range(1,cfg['updates']+1):
        idx=ix[rng.integers(len(ix),size=cfg['batch_tokens'])]
        xb=torch.from_numpy(np.array(x[idx],copy=True)).to(device);yb=torch.from_numpy(y[idx].astype(np.int64)).to(device)
        xb=xb+torch.randn_like(xb)*cfg['noise_aug']*xb.std().detach().clamp_min(1e-4)
        with torch.autocast(device.split(':')[0],dtype=torch.bfloat16):loss=F.cross_entropy(model(xb),yb)
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite affine fitting loss at seed {seed}, update {step}')
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%256==0:print('fit',seed,step,'loss',float(loss),'seconds',round(time.monotonic()-started,1),flush=True)
        if step in cfg['validation_updates']:
            if not all(torch.isfinite(t).all() for t in model.parameters()):
                raise RuntimeError(f'Nonfinite affine parameters at seed {seed}, update {step}')
            model.eval();pred=readout_predictions(model,v,device);model.train()
            agreement=float(((pred==yv)&valid).sum()/valid.sum())
            history.append(dict(step=step,lexical_agreement=agreement,all_position_agreement=float((pred==yv).mean())))
            if agreement>best:
                best=agreement;best_step=step
                torch.save({k:t.detach().cpu() for k,t in model.state_dict().items()},dest/'selected.pt')
            dump(dest/'validation_history.json',history)
        if step%32==0:time.sleep(0.02)
    dump(dest/'selected.json',dict(seed=seed,selected_step=best_step,validation_agreement=best,
        sha256=digest(dest/'selected.pt'),protocol_sha256=digest(OUT/'protocol.json'),elapsed=time.monotonic()-started,
        device=device,train_lexical_tokens=len(ix),torch=torch.__version__))

def source_mask(y):
    valid=y!=50256
    return valid

def make_groups():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2',local_files_only=True)
    y=np.load(OUT/'data/training/y.npy');count=np.bincount(y[y!=50256],minlength=50257)
    numeric=np.array([any(c.isdigit() for c in tok.decode([i],clean_up_tokenization_spaces=False)) for i in range(50257)])
    np.save(OUT/'counts.npy',count);np.save(OUT/'numeric.npy',numeric)

def prepare_outputs(d,device,condition='calibration',gseed=61022301):
    verify_protocol()
    if not (OUT/'counts.npy').exists():make_groups()
    y=np.load(d/'y.npy');v=source_mask(y);margin=np.load(d/'margin.npy')
    for seed in FITS:
        fp=d/f'linear{seed}.npy'
        if fp.exists():continue
        model=load_readout(seed,device);pred=readout_predictions(model,d,device);np.save(fp,pred)
        del model;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    if (d/'repairs.npz').exists():return
    pred=np.load(d/'linear17.npy');counts=np.load(OUT/'counts.npy');numeric=np.load(OUT/'numeric.npy')
    repaired={k:pred.copy() for k in POLICIES}
    for i in range(len(y)):
        errors=np.flatnonzero((pred[i]!=y[i])&v[i]);n=len(errors)//2
        for pi,k in enumerate(POLICIES):
            # Same tie RNG across conditions; paired bootstrap preserves this coupling.
            rng=np.random.default_rng(np.random.SeedSequence([gseed,i,pi,61022]));tie=rng.random(len(errors))
            if k.startswith('random'):chosen=rng.permutation(errors)[:n]
            else:
                score=counts[y[i,errors]] if k=='rare' else -numeric[y[i,errors]].astype(int) if k=='numeric' else margin[i,errors]
                chosen=errors[np.lexsort((tie,score))[:n]]
            repaired[k][i,chosen]=y[i,chosen]
            assert ((repaired[k][i]!=y[i])&v[i]).sum()==len(errors)-n
    temporary=d/'repairs.npz.tmp'
    with temporary.open('wb') as handle:np.savez(handle,**repaired)
    temporary.replace(d/'repairs.npz')

def features(y,pred,margin,native_nll,rare_cutoff=100):
    v=source_mask(y);e=(pred!=y)&v;den=v.sum(1).clip(1)
    c=np.load(OUT/'counts.npy');numeric=np.load(OUT/'numeric.npy')
    return np.stack([e.sum(1)/den,np.log(den),native_nll,
        (e&(c[y]<rare_cutoff)).sum(1)/den,(e&numeric[y]).sum(1)/den,(e&(margin<2)).sum(1)/den],1)

def fit_ridge(x,y):
    mu=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-12,sd,1.)
    z=np.c_[np.ones(len(x)),(x-mu)/sd];reg=np.eye(z.shape[1])*10.;reg[0,0]=0
    beta=np.linalg.solve(z.T@z+reg,z.T@y)
    return dict(mean=mu.tolist(),std=sd.tolist(),beta=beta.tolist())

def predict(x,fit):return np.c_[np.ones(len(x)),(x-np.array(fit['mean']))/np.array(fit['std'])]@np.array(fit['beta'])

def freeze_test():
    verify_protocol()
    if (OUT/'test_frozen.json').exists():verify_test_freeze();return
    if (OUT/'data/test').exists():raise RuntimeError('Test generation must follow freeze')
    d=OUT/'data/calibration';y=np.load(d/'y.npy');margin=np.load(d/'margin.npy')
    scores={}
    for method in ['native']+[f'linear{s}' for s in FITS]+[f'repair_{k}' for k in POLICIES]:
        rows=csvread(d/f'scores_gpt2-large_{method}.csv')
        assert len(rows)==len(y) and [int(r['sequence']) for r in rows]==list(range(len(y)))
        scores[method]=np.array([float(r['nll_sum'])/int(r['tokens']) if int(r['tokens'])>0 else np.nan for r in rows])
    good=np.logical_and.reduce([np.isfinite(x) for x in scores.values()])
    if good.sum()<100:raise RuntimeError('Too few common scorable calibration sequences; report feasibility failure')
    native=scores['native'];fits={};records=[]
    for s in FITS:
        pred=np.load(d/f'linear{s}.npy');x=features(y,pred,margin,native);target=scores[f'linear{s}']-native
        fits[str(s)]=dict(baseline=fit_ridge(x[good,:3],target[good]),tail=fit_ridge(x[good],target[good]))
        cutoff=100*float(np.load(OUT/'counts.npy').sum())/protocol()['frequency_sensitivity']['elf_reference_tokens']
        xf=features(y,pred,margin,native,rare_cutoff=cutoff)
        fits[str(s)]['tail_frequency_matched']=fit_ridge(xf[good],target[good])
        fits[str(s)]['frequency_matched_count_cutoff']=cutoff
        for i in np.flatnonzero(good):records.append(dict(fit_seed=s,sequence=int(i),target=float(target[i]),**{k:float(x[i,j]) for j,k in enumerate(FEATURES)}))
    dump(OUT/'calibration_eligibility.json',dict(total=len(y),common_scorable=int(good.sum()),excluded_indices=np.flatnonzero(~good).tolist()))
    dump(OUT/'predictors.json',fits);csvwrite(OUT/'calibration.csv',records)
    paths=[OUT/'protocol.json',OUT/'predictors.json',OUT/'calibration.csv',OUT/'counts.npy',OUT/'numeric.npy',
        Path(__file__),ROOT/'scripts/aurora_runtime_20261022.py',ROOT/'scripts/aurora_score_20261022.py']
    paths += [ROOT/'scripts/aurora_analyze_20261022.py',ROOT/'scripts/aurora_verify_20261022.py']
    paths += [OUT/'source/model_assets.json',OUT/'source/environment.json']
    paths += [OUT/f'fits/seed{s}/selected.pt' for s in FITS]
    paths += [OUT/'calibration_eligibility.json',OUT/'data/calibration/ready.json']
    paths += sorted((OUT/'data/calibration').glob('scores_gpt2-large_*.csv'))
    dump(OUT/'test_frozen.json',dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),files={str(f.relative_to(ROOT)):digest(f) for f in paths},test_generated=False))
    print('Readouts, predictors and test protocol frozen before test generation',flush=True)

def verify_test_freeze():
    f=json.loads((OUT/'test_frozen.json').read_text())
    for path,sha in f['files'].items():assert digest(ROOT/path)==sha,f'Frozen input changed: {path}'

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['protocol','collect','fit','outputs','freeze-test'])
    p.add_argument('--split',default='training',choices=['training','validation','calibration','test'])
    p.add_argument('--condition',choices=['S16','S32','S64']);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda:2');p.add_argument('--batch',type=int,default=8)
    p.add_argument('--memory-gib',type=float,default=3.5)
    p.add_argument('--range-start',type=int,default=0);p.add_argument('--range-end',type=int);a=p.parse_args()
    if a.mode=='protocol':freeze_protocol(a.batch)
    elif a.mode=='collect':collect(a.split,a.device,a.condition,a.seed,a.memory_gib,a.range_start,a.range_end)
    elif a.mode=='fit':
        from contextlib import nullcontext
        ctx=torch.cuda.stream(torch.cuda.Stream(device=a.device,priority=1)) if a.device.startswith('cuda') else nullcontext()
        with ctx:fit(a.seed,a.device)
    elif a.mode=='outputs':
        torch.set_num_threads(4)
        from contextlib import nullcontext
        if a.device.startswith('cuda'):
            torch.cuda.set_device(a.device);free,total=torch.cuda.mem_get_info(a.device)
            if free<4*2**30:raise RuntimeError('Output generation needs safe free headroom')
            torch.cuda.set_per_process_memory_fraction(2.0*2**30/total,a.device)
        ctx=torch.cuda.stream(torch.cuda.Stream(device=a.device,priority=1)) if a.device.startswith('cuda') else nullcontext()
        with ctx:prepare_outputs(data_dir(a.split,a.condition,a.seed),a.device,a.condition or a.split,a.seed)
    else:freeze_test()

if __name__=='__main__':main()
