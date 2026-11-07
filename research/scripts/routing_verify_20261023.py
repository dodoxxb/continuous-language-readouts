"""Independent portable verification: NumPy/stdlib only, no producer imports.

Checks development refits, fresh predictions, all fixed-budget choices,
analytical random expectation, paired intervals and document nonoverlap.
Does not rerun pretrained models, token feature extraction or raw generation.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np

FEATURES=['mean_log1p_gap','q10_log1p_gap','mean_entropy','mean_log1p_predicted_count',
 'predicted_numeric_fraction','own_eos_present','own_stop_fraction','canvas_length']
FAMILIES=['elf','aurora'];SEEDS=[61032301,61034301]
POLICIES=['ridge','threefit_disagreement','negative_mean_log1p_gap','stopping_only']

def readcsv(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)

def refit(x,y):
 mean=np.mean(x,axis=0);sd=np.std(x,axis=0);sd[sd<=1e-12]=1.
 design=np.column_stack([np.ones(len(x)),(x-mean)/sd]);ridge=np.zeros((len(mean)+1,len(mean)+1));ridge[1:,1:]=10*np.eye(len(mean))
 beta=np.linalg.solve(design.T@design+ridge,design.T@y)
 return mean,sd,beta

def infer(x,m):return np.column_stack([np.ones(len(x)),(x-np.asarray(m['mean']))/np.asarray(m['std'])])@np.asarray(m['beta'])

def ci(x):
 x=np.asarray(x);rng=np.random.default_rng(61032329);bs=x[rng.integers(0,len(x),(4000,len(x)))].mean(axis=1)
 return float(x.mean()),np.quantile(bs,[.025,.975])

def verify(root):
 identities=json.loads((root/'cohort/identity.json').read_text());new=identities['new'];old=identities['previously_observed']
 assert len(new)==256 and len({r['id'] for r in new})==256 and len({r['document_sha256'] for r in new})==256
 assert not {r['id'] for r in new}&{r['id'] for r in old}
 assert not {r['document_sha256'] for r in new}&{r['document_sha256'] for r in old}
 ids={int(r['index']):r for r in new};assert set(ids)==set(range(256))
 protocol=json.loads((root/'protocol/protocol.json').read_text());assert protocol['test']['generation_seeds']==SEEDS
 weights=json.loads((root/'calibration/weights.json').read_text());models=weights['models']
 fit=readcsv(root/'calibration/fit_data.csv');assert len(fit)==4096
 assert sha(root/'calibration/fit_data.csv')==weights['fit_data_sha256']
 refit_checks=[]
 for family in FAMILIES:
  r=[q for q in fit if q['family']==family];assert len(r)==2048
  assert len({q['document_id'] for q in r})==1024
  assert not {q['document_id'] for q in r}&{q['id'] for q in new}
  x=np.array([[float(q[k]) for k in FEATURES] for q in r]);y=np.array([float(q['target_native_minus_autonomous_bertscore_points']) for q in r])
  for label,xx,m in [('full',x,models[family]),('stopping_only',x[:,5:8],models[family]['stopping_only'])]:
   mean,sd,beta=refit(xx,y);close(mean,m['mean']);close(sd,m['std']);close(beta,m['beta']);assert m['alpha']==10.
   refit_checks.append(f'{family}:{label}')
 rows=readcsv(root/'analysis/scalar_inputs.csv');assert len(rows)==1024
 index={(r['family'],int(r['generation_seed']),int(r['index'])):r for r in rows};assert len(index)==1024
 for family in FAMILIES:
  for seed in SEEDS:
   r=[index[(family,seed,i)] for i in range(256)]
   for q in r:
    identity=ids[int(q['index'])];assert q['document_id']==identity['id'] and q['document_sha256']==identity['document_sha256']
    assert q['reference_sha256']==identity['reference_sha256']
   x=np.array([[float(q[k]) for k in FEATURES] for q in r])
   close(infer(x,models[family]),[float(q['risk_ridge']) for q in r])
   close(infer(x[:,5:8],models[family]['stopping_only']),[float(q['risk_stopping_only']) for q in r])
   close(-x[:,0],[float(q['risk_negative_mean_log1p_gap']) for q in r])
   for policy in POLICIES:
    def key(i):
     h=hashlib.sha256(f"{family}|{seed}|{r[i]['document_id']}".encode()).hexdigest()
     return -float(r[i]['risk_'+policy]),h,i
    chosen=set(sorted(range(256),key=key)[:128]);actual=np.array([int(q['route_'+policy]) for q in r])
    assert actual.sum()==128 and np.array_equal(actual,[int(i in chosen) for i in range(256)])
 def array(k):return np.array([[[float(index[(f,s,i)][k]) for s in SEEDS] for f in FAMILIES] for i in range(256)])
 Bn,Ba=array('native_bertscore'),array('autonomous17_bertscore');Nn,Na=array('native_nll'),array('autonomous17_nll')
 eligible=((array('native_nll_eligible')>0)&(array('autonomous17_nll_eligible')>0)).all((1,2))
 result=json.loads((root/'analysis/summary.json').read_text());assert result['status']=='complete'
 assert result['scalar_inputs_sha256']==sha(root/'analysis/scalar_inputs.csv')
 assert result['nll_cohort']['excluded_indices']==np.flatnonzero(~eligible).tolist()
 assert result['nll_cohort']['documents']==int(eligible.sum()) and result['nll_cohort']['comparator_probability']==.5
 checks=[]
 for policy in POLICIES:
  G=array('route_'+policy);bd=(G-.5)*(Bn-Ba);nd=(G-.5)*(Na-Nn)
  for name,f in [('pooled',None),('elf',0),('aurora',1)]:
   b=bd.mean((1,2)) if f is None else bd[:,f,:].mean(1)
   point,bounds=ci(b);saved=result['policies'][policy]['bertscore'][name]
   close(point,saved['point']);close(bounds,saved['ci95']);assert saved['documents']==256
   selected=Ba+G*(Bn-Ba);random=.5*(Bn+Ba)
   close(selected.mean() if f is None else selected[:,f,:].mean(),saved['routed_score'])
   close(random.mean() if f is None else random[:,f,:].mean(),saved['random_expected_score'])
   close(saved['routed_score']-saved['random_expected_score'],point)
   checks.append(f'{policy}:BERT:{name}')
   if eligible.any():
    n=nd[eligible].mean((1,2)) if f is None else nd[eligible,f,:].mean(1)
    point,bounds=ci(n);saved=result['policies'][policy]['nll'][name]
    close(point,saved['point']);close(bounds,saved['ci95'])
    close(saved['random_expected_score']-saved['routed_score'],point)
    close(G[eligible].mean() if f is None else G[eligible,f,:].mean(),saved['actual_subset_native_call_fraction'])
    checks.append(f'{policy}:NLL:{name}')
  for family in FAMILIES:
   rr=[r for r in rows if r['family']==family];g=np.array([int(r['route_'+policy]) for r in rr]);ms=np.array([float(r['native_terminal_forward_cuda_ms']) for r in rr])
   saved=result['policies'][policy]['descriptive_cost_proxy'][family]
   close(np.mean(g*ms),saved['selected_native_forward_ms_per_output']);close(.5*ms.mean(),saved['random_expected_native_forward_ms_per_output'])
 Gr,Gs=array('route_ridge'),array('route_stopping_only')
 for name,f in [('pooled',None),('elf',0),('aurora',1)]:
  v=(Gr-Gs)*(Bn-Ba);x=v.mean((1,2)) if f is None else v[:,f,:].mean(1)
  point,bounds=ci(x);saved=result['stopping_ablation'][name]['bertscore_full_minus_stopping'];close(point,saved['point']);close(bounds,saved['ci95'])
  if eligible.any():
   v=(Gr-Gs)*(Na-Nn);x=v[eligible].mean((1,2)) if f is None else v[eligible,f,:].mean(1)
   point,bounds=ci(x);saved=result['stopping_ablation'][name]['nll_benefit_full_minus_stopping'];close(point,saved['point']);close(bounds,saved['ci95'])
 checked_hashes={};omitted=[];frozen=json.loads((root/'frozen.json').read_text())
 prefix='outputs/native_free_routing_20261023/'
 for relative,h in frozen['files'].items():
  p=root/relative[len(prefix):] if relative.startswith(prefix) else root/'code'/Path(relative).name
  if p.exists():assert sha(p)==h,relative;checked_hashes[str(p.relative_to(root))]=h
  else:omitted.append(relative)
 return dict(status='passed',documents=256,generations=1024,development_refits=refit_checks,
  verified_effect_intervals=checks,all_four_policy_budgets_passed=True,random_expectation_probability=.5,
  overlap_check='New versus original test/pilot IDs and exact document hashes passed; development IDs disjoint',
  checked_frozen_files=checked_hashes,not_present_for_hash_verification=omitted,
  scope_limitations=['Does not regenerate pretrained outputs or recalculate neural metric scores.',
   'Scalar feature extraction is preserved/provenanced but is not rerun from large state arrays here.',
   'Full train-corpus overlap was checked at cohort preparation against pinned parquet; this portable verifier checks supplied old/new identity lists.',
   'Confidence intervals condition on fitted models and fixed fresh routing decisions; same calls are not same measured runtime.'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-root',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/native_free_routing_20261023');a=p.parse_args()
 result=verify(a.input_root);dest=a.input_root/'analysis/verification.json';dest.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['checked_frozen_files','not_present_for_hash_verification']},indent=2))

if __name__=='__main__':main()
