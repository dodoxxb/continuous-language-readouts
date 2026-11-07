"""Post-result control: displace only positions native-valid in both sequences."""
from pathlib import Path
from types import SimpleNamespace
import argparse,importlib.util,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('frozen_context',ROOT/'scripts/context_boundary_20261021.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
OUT=old.OUT/'valid_followup';PROTOCOL=OUT/'protocol.json'
def freeze():
 assert not PROTOCOL.exists()
 p=dict(date='2026-10-22',design_status='Post-result follow-up designed after the full-canvas context results. Frozen before these follow-up outcomes.',motivation='Does the donor effect require editing positions after EOS?',parent_protocol_sha256=old.sha(old.OUT/'context_protocol.json'),script_sha256=old.sha(__file__),conditions=list(old.CONDITIONS),seeds=old.SEEDS,sequences_per_seed=64,probes_per_sequence=8,mask='Intersection of original and cyclic-donor native-valid positions, excluding all eight original protected probes. All other canvas vectors are unchanged.',arms=['fresh baseline','independent zero control','full donor displacement on shared-valid context','Gaussian on same mask, matched to donor Frobenius norm'],gaussian_seed='SeedSequence([61022, condition_index, generation_seed, sequence_index])',analysis='Same 4000 sequence-cluster bootstrap draws stratified by generation seed; descriptive conditional 95% intervals. All conditions and both nonzero arms retained.')
 old.dump(PROTOCOL,p);print(old.sha(PROTOCOL),flush=True)
def run(cond,device):
 import torch
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28,device)
 p=json.loads(PROTOCOL.read_text());assert old.sha(__file__)==p['script_sha256']
 parent=json.loads((old.OUT/'context_protocol.json').read_text());assert old.sha(old.OUT/'context_protocol.json')==p['parent_protocol_sha256'];assert old.sha(ROOT/'scripts/context_boundary_20261021.py')==parent['files']['scripts/context_boundary_20261021.py']
 modelname,cp,cfg=old.CONDITIONS[cond];md=ROOT/'models'/modelname
 for q in [md/cp,md/'config.yml']:assert old.sha(q)==parent['files'][str(q.relative_to(ROOT))]
 sys.path.insert(0,str(ROOT/'outputs/phase0_phase1_20260520/scripts'))
 from phase2_trajectory_audit import load_elf,decode_logits
 args=SimpleNamespace(root=ROOT,out=OUT/cond,encoder_model=ROOT/'models/t5-small',tokenizer=ROOT/'models/t5-small',ppl_model=ROOT/'models/gpt2-large',config=md/'config.yml',checkpoint=md/cp,weights='ema_params1',num_samples=64,batch_size=1)
 config,tok,model=load_elf(args,torch.device(device));model.eval();ci=list(old.CONDITIONS).index(cond);started=time.time()
 with torch.inference_mode():
  for seed in old.SEEDS:
   src=old.OLD/'test'/cond/str(seed)
   for n in ['x.npy','tokens.npz']:assert old.sha(src/n)==parent['files'][str((src/n).relative_to(ROOT))]
   xs=np.load(src/'x.npy',mmap_mode='r');pack=np.load(src/'tokens.npz')
   for i in range(64):
    dest=OUT/cond/str(seed)/f'{i:03d}.json'
    if dest.exists():continue
    idx=(i+1)%64;valid=np.flatnonzero(pack['valid'][i]);anchors=valid[np.rint(np.linspace(0,len(valid)-1,8)).astype(int)]
    mask=pack['valid'][i].astype(bool)&pack['valid'][idx].astype(bool);mask[anchors]=False;assert mask.any()
    x=torch.from_numpy(np.array(xs[i],copy=True)).to(device);donor=np.array(xs[idx]-xs[i]);donor[~mask]=0
    rng=np.random.default_rng(np.random.SeedSequence([61022,ci,seed,i]));gaussian=rng.standard_normal(x.shape,dtype=np.float32);gaussian[~mask]=0;gaussian*=np.linalg.norm(donor)/np.linalg.norm(gaussian)
    a=torch.as_tensor(anchors,device=device);unchanged=torch.as_tensor(~mask,device=device);base=decode_logits(model,x[None],1.,cfg,config)[0,a].float();target=base.argmax(-1)
    prior=json.loads((old.OUT/'context'/cond/str(seed)/f'{i:03d}.json').read_text());original=[r for r in prior['rows'] if r['kind']=='zero'];assert target.tolist()==[r['native_base'] for r in original]
    rows=[];invariants=[]
    for kind,alpha,d in [('zero',0.,donor),('donor',1.,donor),('gaussian',1.,gaussian)]:
     xx=x+alpha*torch.from_numpy(d).to(device);assert torch.equal(xx[unchanged],x[unchanged])
     logits=decode_logits(model,xx[None],1.,cfg,config)[0,a].float()
     if kind=='zero':assert torch.equal(base,logits)
     arg=logits.argmax(-1)
     for j,pos in enumerate(anchors):rows.append(dict(condition=cond,generation_seed=seed,sequence=i,probe=j,position=int(pos),kind=kind,native_base=int(target[j]),native_perturbed=int(arg[j]),flip=int(target[j]!=arg[j]),changed_context_positions=int(mask.sum()),relative_displacement=float(torch.linalg.vector_norm(xx-x)/torch.linalg.vector_norm(x))))
     invariants.append(dict(kind=kind,protected_and_outside_mask_max_delta=float((xx[unchanged]-x[unchanged]).abs().max())))
    old.dump(dest,dict(rows=rows,invariants=invariants,donor_norm=float(np.linalg.norm(donor)),gaussian_norm=float(np.linalg.norm(gaussian)),protocol_sha256=old.sha(PROTOCOL)))
    if i%32==0:print(cond,seed,i,round(time.time()-started,1),flush=True)
 print('Complete',cond,round(time.time()-started,1),flush=True)
def analyze():
 rows=[];invariants=[]
 for c in old.CONDITIONS:
  for seed in old.SEEDS:
   for i in range(64):
    d=json.loads((OUT/c/str(seed)/f'{i:03d}.json').read_text());assert d['protocol_sha256']==old.sha(PROTOCOL);rows+=d.pop('rows');invariants.append(dict(condition=c,generation_seed=seed,sequence=i,**d))
 old.writecsv(OUT/'records.csv',rows);old.dump(OUT/'invariants.json',invariants);summary=[]
 for c in old.CONDITIONS:
  for k in ['zero','donor','gaussian']:
   rr=[r for r in rows if r['condition']==c and r['kind']==k]
   x=np.array([[sum(r['flip'] for r in rr if r['generation_seed']==seed and r['sequence']==i)/8 for i in range(64)] for seed in old.SEEDS]);rng=np.random.default_rng(61021);boot=sum(x[j][rng.integers(64,size=(4000,64))].mean(1)/2 for j in range(2));lo,hi=np.quantile(boot,[.025,.975])
   summary.append(dict(condition=c,kind=k,flip_rate=float(x.mean()),ci_low=float(lo),ci_high=float(hi),mean_relative_displacement=float(np.mean([r['relative_displacement'] for r in rr])),mean_changed_context_positions=float(np.mean([r['changed_context_positions'] for r in rr]))))
 old.writecsv(OUT/'summary.csv',summary);old.dump(OUT/'analysis.json',dict(sequences=384,probes=3072,native_forwards=1536,records=len(rows),all_original_baselines_matched=True,zero_control_flips=sum(r['flip'] for r in rows if r['kind']=='zero')));print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze']);p.add_argument('--condition',choices=old.CONDITIONS);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.action=='freeze':freeze()
 elif a.action=='run':run(a.condition,a.device)
 else:analyze()
