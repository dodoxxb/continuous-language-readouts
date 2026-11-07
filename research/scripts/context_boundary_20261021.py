"""Frozen-state context intervention: keep probe latents exactly unchanged.

These are synthetic decoder stress tests on reused states, not new generation,
task-quality measurements, or a trial of a trained commitment architecture.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,csv,hashlib,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/boundary_audit_20261021'
OLD=ROOT/'outputs/prospective_tail_20261013'
CONDITIONS={'B32':('ELF-B-owt-torch','checkpoint_95085',3.),'M32':('ELF-M-owt-torch','checkpoint_76070',3.),'PD8':('ELF-B-PD-owt-torch','checkpoint_19017',2.5)}
SEEDS=[61013201,61013202]
LEVELS=[.1,.25,.5,1.]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def writecsv(p,rows):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def freeze():
 if (OUT/'context_protocol.json').exists():raise RuntimeError('Protocol already frozen; do not overwrite')
 files={str(Path(__file__).resolve().relative_to(ROOT)):sha(__file__)}
 for cond,(model,checkpoint,cfg) in CONDITIONS.items():
  for p in [ROOT/'models'/model/checkpoint,ROOT/'models'/model/'config.yml']:
   files[str(p.relative_to(ROOT))]=sha(p)
  for seed in SEEDS:
   for n in ['x.npy','tokens.npz']:
    p=OLD/'test'/cond/str(seed)/n;files[str(p.relative_to(ROOT))]=sha(p)
 head=ROOT/'outputs/confirmatory_readout_20261013/fits/elf/n16384_seed17/selected.pt';files[str(head.relative_to(ROOT))]=sha(head)
 p=dict(date='2026-10-21',scope='New protocol-frozen interventions on previously analyzed endpoint states; not prospective new-generation tests.',question='Does changing only context change a native decoder decision when the probed latent and its token-wise affine decision are fixed?',conditions=CONDITIONS,generation_seeds=SEEDS,sequence_indices=list(range(64)),probes_per_sequence=8,probe_selection='Eight equally spaced native-valid positions, rounding linspace indices; no selection by confidence, error or outcome.',donor='Next sequence cyclically within each 64-sequence condition/seed subset.',perturbations=['donor displacement outside the eight protected positions','Gaussian displacement outside the same positions, normalized to the donor displacement Frobenius norm'],amplitudes=LEVELS,gaussian_rng='NumPy default_rng(SeedSequence([61021, condition_index, generation_seed, sequence_index])); float32 normal array.',mask='Change the full decoder canvas outside probes, including positions after native EOS; evaluate only the original eight valid probes.',control='Fresh deterministic native baseline plus a separate zero-displacement forward pass; bitwise equality of protected input vectors; exact token-wise affine decisions across all interventions.',readout_fit_seed=17,evaluation='Native argmax changes and reference margins at protected positions; no evaluator NLL or task-accuracy interpretation.',inference='4000 sequence-cluster bootstrap draws seed 61021, stratified by original generation seed, 64 sequences per stratum. Probe positions averaged within a sequence. Conditional 95% descriptive intervals; no denoiser-training-seed inference.',primary_display='All prespecified amplitudes and both directions for all conditions; no selection of favorable condition or direction.',files=files)
 dump(OUT/'context_protocol.json',p);print('Frozen protocol',sha(OUT/'context_protocol.json'),flush=True)
def run(cond,device):
 import torch
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.28,device)
 protocol=json.loads((OUT/'context_protocol.json').read_text())
 assert sha(__file__)==protocol['files'][str(Path(__file__).resolve().relative_to(ROOT))]
 modelname,checkpoint,cfg=CONDITIONS[cond];md=ROOT/'models'/modelname
 for p in [md/checkpoint,md/'config.yml']:
  assert sha(p)==protocol['files'][str(p.relative_to(ROOT))]
 sys.path.insert(0,str(ROOT/'outputs/phase0_phase1_20260520/scripts'))
 from phase2_trajectory_audit import load_elf,decode_logits
 args=SimpleNamespace(root=ROOT,out=OUT/'context'/cond,encoder_model=ROOT/'models/t5-small',tokenizer=ROOT/'models/t5-small',ppl_model=ROOT/'models/gpt2-large',config=md/'config.yml',checkpoint=md/checkpoint,weights='ema_params1',num_samples=64,batch_size=1)
 c,tok,model=load_elf(args,torch.device(device));model.eval()
 hp=ROOT/'outputs/confirmatory_readout_20261013/fits/elf/n16384_seed17/selected.pt'
 assert sha(hp)==protocol['files'][str(hp.relative_to(ROOT))]
 weights=torch.load(hp,map_location=device,weights_only=True)
 head=torch.nn.Linear(weights['weight'].shape[1],weights['weight'].shape[0]).to(device);head.load_state_dict(weights);head.eval()
 started=time.time();ci=list(CONDITIONS).index(cond)
 with torch.inference_mode():
  for seed in SEEDS:
   src=OLD/'test'/cond/str(seed)
   for name in ['x.npy','tokens.npz']:assert sha(src/name)==protocol['files'][str((src/name).relative_to(ROOT))]
   xs=np.load(src/'x.npy',mmap_mode='r');pack=np.load(src/'tokens.npz')
   for i in range(64):
    dest=OUT/'context'/cond/str(seed)/(f'{i:03d}.json')
    if dest.exists():continue
    valid=np.flatnonzero(pack['valid'][i]);assert len(valid)>=8
    anchors=valid[np.rint(np.linspace(0,len(valid)-1,8)).astype(int)];assert len(np.unique(anchors))==8
    x=torch.from_numpy(np.array(xs[i],copy=True)).to(device);donor=np.array(xs[(i+1)%64]-xs[i]);donor[anchors]=0
    rng=np.random.default_rng(np.random.SeedSequence([61021,ci,seed,i]));gaussian=rng.standard_normal(x.shape,dtype=np.float32);gaussian[anchors]=0;gaussian*=np.linalg.norm(donor)/np.linalg.norm(gaussian)
    dnorm=float(np.linalg.norm(donor));gnorm=float(np.linalg.norm(gaussian));assert np.isclose(dnorm,gnorm,rtol=2e-6)
    a=torch.as_tensor(anchors,device=device);anchor=x[a].clone();linear=head(anchor).argmax(-1)
    base=decode_logits(model,x[None],1.,cfg,c)[0,a].float();top=base.topk(2,-1);target=top.indices[:,0];margin=top.values[:,0]-top.values[:,1]
    rows=[];invariants=[]
    for kind,alpha,delta in [('zero',0.,donor)]+[(k,v,d) for k,d in [('donor',donor),('gaussian',gaussian)] for v in LEVELS]:
     xx=x+alpha*torch.from_numpy(delta).to(device)
     assert torch.equal(xx[a],anchor),'Protected state changed'
     lin=head(xx[a]).argmax(-1);assert torch.equal(lin,linear)
     logits=decode_logits(model,xx[None],1.,cfg,c)[0,a].float();arg=logits.argmax(-1)
     others=logits.clone();others[torch.arange(8,device=device),target]=-torch.inf
     refmargin=logits[torch.arange(8,device=device),target]-others.max(-1).values
     if kind=='zero':assert torch.equal(base,logits),'Nondeterministic zero-displacement control'
     for j,pos in enumerate(anchors):
      rows.append(dict(condition=cond,generation_seed=seed,sequence=i,probe=j,position=int(pos),kind=kind,amplitude=alpha,native_base=int(target[j]),native_perturbed=int(arg[j]),cached_native=int(pack['native'][i,pos]),linear_token=int(linear[j]),native_margin=float(margin[j]),reference_margin=float(refmargin[j]),flip=int(target[j]!=arg[j]),linear_flip=int(linear[j]!=lin[j]),native_linear_disagreement=int(linear[j]!=target[j]),context_delta_norm=float(torch.linalg.vector_norm(xx-x)),base_norm=float(torch.linalg.vector_norm(x))))
     invariants.append(dict(kind=kind,amplitude=alpha,protected_max_absolute_delta=float((xx[a]-anchor).abs().max()),affine_changes=int((lin!=linear).sum())))
    dump(dest,dict(rows=rows,invariants=invariants,donor_norm=dnorm,gaussian_norm=gnorm,protocol_sha256=sha(OUT/'context_protocol.json')))
    if i%16==0:print(cond,seed,i,'elapsed',round(time.time()-started,1),flush=True)
 print('Completed',cond,'seconds',round(time.time()-started,1),flush=True)
def analyze():
 rows=[]
 for cond in CONDITIONS:
  for seed in SEEDS:
   for i in range(64):
    d=json.loads((OUT/'context'/cond/str(seed)/(f'{i:03d}.json')).read_text());assert d['protocol_sha256']==sha(OUT/'context_protocol.json')
    assert all(x['protected_max_absolute_delta']==0 and x['affine_changes']==0 for x in d['invariants']);rows+=d['rows']
 writecsv(OUT/'context_records.csv',rows)
 summary=[];seqrows=[]
 for cond in CONDITIONS:
  for kind,alpha in [('zero',0.)]+[(k,a) for k in ['donor','gaussian'] for a in LEVELS]:
   rr=[r for r in rows if r['condition']==cond and r['kind']==kind and r['amplitude']==alpha]
   x=np.array([np.mean([r['flip'] for r in rr if r['generation_seed']==seed and r['sequence']==i]) for seed in SEEDS for i in range(64)]).reshape(2,64)
   rng=np.random.default_rng(61021);boot=sum(x[j][rng.integers(64,size=(4000,64))].mean(1)/2 for j in range(2));lo,hi=np.quantile(boot,[.025,.975])
   summary.append(dict(condition=cond,kind=kind,amplitude=alpha,sequences=128,probes=1024,flip_rate=float(x.mean()),ci_low=float(lo),ci_high=float(hi),mean_relative_context_displacement=float(np.mean([r['context_delta_norm']/r['base_norm'] for r in rr]))))
   for j,seed in enumerate(SEEDS):
    for i in range(64):seqrows.append(dict(condition=cond,generation_seed=seed,sequence=i,kind=kind,amplitude=alpha,flip_rate=float(x[j,i])))
 writecsv(OUT/'context_summary.csv',summary);writecsv(OUT/'context_sequence.csv',seqrows)
 baseline=[r for r in rows if r['kind']=='zero'];scales=[]
 for cond in CONDITIONS:
  rr=[r for r in baseline if r['condition']==cond]
  for scale in [.5,1.,2.]:scales.append(dict(condition=cond,logit_scale=scale,margin8_fraction=float(np.mean([scale*r['native_margin']>=8 for r in rr])),native_decisions_changed=0,interpretation='Exact algebraic positive scaling of saved logits/margins, not new sampling'))
 writecsv(OUT/'logit_scale_control.csv',scales)
 dump(OUT/'context_analysis.json',dict(status='passed',sequences=384,probe_instances=3072,records=len(rows),native_forwards=384*10,all_protected_inputs_unchanged=True,all_affine_decisions_unchanged=True,zero_control_flips=sum(r['flip'] for r in baseline),cached_target_mismatches=sum(r['native_base']!=r['cached_native'] for r in baseline),summary=summary))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze']);p.add_argument('--condition',choices=CONDITIONS);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 if a.action=='freeze':freeze()
 elif a.action=='run':run(a.condition,a.device)
 else:analyze()
