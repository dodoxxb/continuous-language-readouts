"""Frozen affine-only sequence risk features and development-only ridge fitting.

Feature extraction reads x/affine logits only; native fields and scores are
never inputs. Native-minus-autonomous BERTScore is a development target only.
"""
from pathlib import Path
from contextlib import nullcontext
import argparse,csv,hashlib,json,os,time
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/native_free_routing_20261023'
OLD=ROOT/'outputs/task_readout_20261022'
FITS=[17,29,43]
FEATURE_NAMES=['mean_log1p_gap','q10_log1p_gap','mean_entropy',
 'mean_log1p_predicted_count','predicted_numeric_fraction','own_eos_present',
 'own_stop_fraction','canvas_length']

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def dump(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def writecsv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def readcsv(path):
 with Path(path).open() as f:return list(csv.DictReader(f))

def own_boundary(pred,eos_id,family):
 start=0
 if family=='aurora':
  while start<len(pred) and pred[start]==eos_id:start+=1
 loc=np.flatnonzero(pred[start:]==eos_id)
 stop=start+int(loc[0]) if len(loc) else len(pred)
 return start,stop

def features_from_logits(logits,counts,numeric,eos_id,family):
 """Pure affine interface. logits is {17,29,43: [target_canvas,V] tensor}.

 All summaries use the entire target canvas, including positions after the
 affine head's own EOS. The two stop features depend only on affine17.
 No focal/native validity mask is accepted by this API.
 """
 import torch
 z=logits[17].float();pred={s:logits[s].argmax(-1).detach().cpu().numpy() for s in FITS}
 assert z.ndim==2 and len(z)>0 and all(len(q)==len(z) for q in pred.values())
 top=z.topk(2,-1).values;lg=torch.log1p((top[:,0]-top[:,1]).clamp_min(0))
 logp=z-torch.logsumexp(z,dim=-1,keepdim=True)
 entropy=-(logp.exp()*logp).sum(-1)
 a=pred[17];_,stop=own_boundary(a,eos_id,family)
 vals=np.array([float(lg.mean()),float(torch.quantile(lg,0.1)),float(entropy.mean()),
  float(np.log1p(counts[a]).mean()),float(numeric[a].mean()),float(np.any(a==eos_id)),
  float(stop/len(a)),float(len(a))],dtype=np.float64)
 disagreement=np.mean([(pred[a]!=pred[b]).mean() for a,b in [(17,29),(17,43),(29,43)]])
 assert np.isfinite(vals).all()
 return vals,dict(threefit_disagreement=float(disagreement),negative_mean_log1p_gap=-float(vals[0])),pred

def assets(family):
 if family=='elf':
  maps=[ROOT/f'outputs/confirmatory_readout_20261013/fits/elf/n16384_seed{s}/selected.pt' for s in FITS]
  count=ROOT/'outputs/prospective_tail_20261013/token_counts.npy'
  numeric=count.with_name('numeric.npy');eos=1
 else:
  maps=[ROOT/f'outputs/aurora_replication_20261022/fits/seed{s}/selected.pt' for s in FITS]
  count=ROOT/'outputs/aurora_replication_20261022/counts.npy'
  numeric=count.with_name('numeric.npy');eos=50256
 return maps,count,numeric,eos

def extract(family,split,device='cpu',seed=None):
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 if device.startswith('cuda'):
  from task_readout_runtime_20261022 import resources
  # resources also sets interop threads, so impose the same cap locally.
  torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
  if free<7.5*2**30:raise RuntimeError('Insufficient 5.5 GiB cap plus 2 GiB headroom')
  torch.cuda.set_per_process_memory_fraction(5.5*2**30/total,device)
 maps,cp,npth,eos=assets(family);counts=np.load(cp);numeric=np.load(npth);models={}
 for s,p in zip(FITS,maps):
  state=torch.load(p,map_location=device,weights_only=True)
  m=torch.nn.Linear(state['weight'].shape[1],state['weight'].shape[0],device=device)
  m.load_state_dict(state);models[s]=m.eval()
 source=OLD if split=='calibration' else OUT
 cohort={r['index']:r for r in map(json.loads,(source/('cohort.jsonl' if split=='calibration' else 'cohort/cohort.jsonl')).read_text().splitlines())}
 paths=sorted((source/'generated'/family).glob(f'{seed if seed else "*"}/[0-9][0-9][0-9][0-9].npz'))
 assert paths
 target=OUT/(f'calibration/features_{family}.csv' if split=='calibration' else f'analysis/features_{family}_{seed if seed else "all"}.csv')
 if target.exists():raise RuntimeError(f'Refusing to overwrite {target}')
 rows=[];started=time.monotonic()
 stream=None
 if device.startswith('cuda'):
  stream=torch.cuda.Stream(device=device,priority=1)
  stream.wait_stream(torch.cuda.current_stream(device))
 with torch.inference_mode():
  for p in paths:
   idx=int(p.stem);gs=int(p.parent.name)
   # Only x is read. This is the known generated target canvas, not a native-stop slice.
   with np.load(p) as z:x=np.array(z['x'],copy=True)
   with torch.cuda.stream(stream) if stream is not None else nullcontext():
    xt=torch.from_numpy(x).to(device=device,dtype=torch.float32)
    logits={s:m(xt) for s,m in models.items()}
    f,secondary,pred=features_from_logits(logits,counts,numeric,eos,family)
   r=dict(family=family,generation_seed=gs,index=idx,document_id=cohort[idx]['id'],
    document_sha256=cohort[idx]['document_sha256'],state_sha256=sha(p),
    affine17_token_sha256=hashlib.sha256(np.asarray(pred[17],dtype='<i4').tobytes()).hexdigest(),
    **{k:float(v) for k,v in zip(FEATURE_NAMES,f)},**secondary)
   rows.append(r);del logits,xt
   if len(rows)%128==0:print(family,split,'features',len(rows),'seconds',round(time.monotonic()-started,1),flush=True)
 writecsv(target,rows)
 dump(target.with_suffix('.json'),dict(status='complete',rows=len(rows),family=family,split=split,
  features=FEATURE_NAMES,full_target_canvas=True,native_inputs_used=False,device=device,
  elapsed_seconds=time.monotonic()-started,csv_sha256=sha(target),
  affine_asset_sha256={str(p.relative_to(ROOT)):sha(p) for p in maps+[cp,npth]},
  code_sha256=sha(Path(__file__))))
 print('complete',target,len(rows),flush=True)

def fit_model(x,y):
 mean=x.mean(0);sd=x.std(0);sd=np.where(sd>1e-12,sd,1.)
 z=np.c_[np.ones(len(x)),(x-mean)/sd];reg=np.eye(z.shape[1])*10.;reg[0,0]=0
 beta=np.linalg.solve(z.T@z+reg,z.T@y)
 return dict(mean=mean.tolist(),std=sd.tolist(),beta=beta.tolist(),alpha=10.)

def predict(x,model):
 return np.c_[np.ones(len(x)),(x-np.array(model['mean']))/np.array(model['std'])]@np.array(model['beta'])

def old_score(family,index,text,cohortrow):
 h=hashlib.sha256(text.encode()).hexdigest()
 p=OLD/f'scores/bertscore/{family}/{index:04}_{h}.json';q=json.loads(p.read_text())
 assert q['output_sha256']==h and q['source_sha256']==cohortrow['document_sha256'] and q['reference_sha256']==cohortrow['reference_sha256']
 return 100.*q['value']['f1'],p

def fit():
 if (OUT/'frozen.json').exists() or (OUT/'generated').exists():raise RuntimeError('Never fit after freeze or fresh generation')
 cohort={r['index']:r for r in map(json.loads,(OLD/'cohort.jsonl').read_text().splitlines())}
 result={};allrows=[];provenance={}
 for family in ['elf','aurora']:
  path=OUT/f'calibration/features_{family}.csv';rows=readcsv(path);assert len(rows)==2048
  assert len({int(r['index']) for r in rows})==1024
  x=np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows]);ys=[]
  for r in rows:
   i=int(r['index']);gs=int(r['generation_seed']);p=OLD/f'texts/{family}/{gs}/{i:04}.json';text=json.loads(p.read_text())['texts']
   native,pn=old_score(family,i,text['native'],cohort[i]);auto,pa=old_score(family,i,text['autonomous17'],cohort[i])
   y=native-auto;ys.append(y);allrows.append(dict(**r,target_native_minus_autonomous_bertscore_points=y))
   for fp in [p,pn,pa]:provenance[str(fp.relative_to(ROOT))]=sha(fp)
  y=np.array(ys);model=fit_model(x,y)
  # Grouped document CV is descriptive on previously observed development.
  ids=sorted({int(r['index']) for r in rows});rng=np.random.default_rng(61032319);rng.shuffle(ids)
  fold={j:k%5 for k,j in enumerate(ids)};oof=np.empty(len(y))
  for f in range(5):
   va=np.array([fold[int(r['index'])]==f for r in rows]);oof[va]=predict(x[va],fit_model(x[~va],y[~va]))
  model.update(features=FEATURE_NAMES,training_sequences=len(y),training_documents=1024,
   training_target='100*(native BERTScore F1 - autonomous17 BERTScore F1), each own stopping boundary',
   task_specific_development=True,no_hyperparameter_search=True,
   development_target_mean=float(y.mean()),development_grouped_oof_mae=float(np.abs(oof-y).mean()),
   development_grouped_oof_constant_mae=float(np.mean([np.abs(y[np.array([fold[int(r['index'])]==f for r in rows])]-y[np.array([fold[int(r['index'])]!=f for r in rows])].mean()).sum() for f in range(5)])/len(y)*5),
   feature_csv_sha256=sha(path))
  stop_model=fit_model(x[:,5:8],y)
  stop_model.update(features=FEATURE_NAMES[5:8],training_sequences=len(y),training_documents=1024,
   training_target=model['training_target'],task_specific_development=True,no_hyperparameter_search=True)
  model['stopping_only']=stop_model
  result[family]=model
 writecsv(OUT/'calibration/fit_data.csv',allrows)
 dump(OUT/'calibration/target_provenance.json',provenance)
 dump(OUT/'calibration/weights.json',dict(status='fitted_before_fresh_generation',models=result,
  feature_names=FEATURE_NAMES,code_sha256=sha(Path(__file__)),fit_data_sha256=sha(OUT/'calibration/fit_data.csv'),
  target_provenance_sha256=sha(OUT/'calibration/target_provenance.json')))
 print('weights_sha256',sha(OUT/'calibration/weights.json'),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','fit']);p.add_argument('--family',choices=['elf','aurora'])
 p.add_argument('--split',choices=['calibration','test'],default='calibration');p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int);a=p.parse_args()
 if a.mode=='fit':fit()
 else:extract(a.family,a.split,a.device,a.seed)

if __name__=='__main__':main()
