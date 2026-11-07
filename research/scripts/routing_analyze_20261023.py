"""Predeclared paired analysis of the independent native-free routing audit."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from routing_features_20261023 import ROOT,OUT,FEATURE_NAMES,predict,readcsv,writecsv,sha,dump

FAMILIES=['elf','aurora'];SEEDS=[61032301,61034301]
POLICIES=['ridge','threefit_disagreement','negative_mean_log1p_gap','stopping_only']

def tie(family,seed,docid):return hashlib.sha256(f'{family}|{seed}|{docid}'.encode()).hexdigest()

def choices(rows,policy):
 order=sorted(range(len(rows)),key=lambda j:(-float(rows[j]['risk_'+policy]),tie(rows[j]['family'],rows[j]['generation_seed'],rows[j]['document_id']),int(rows[j]['index'])))
 assert len(rows)==256
 selected=set(order[:128]);return [int(j in selected) for j in range(len(rows))]

def metric(family,index,th,metricname,cohortrow):
 p=OUT/f'scores/{metricname}/{family}/{index:04}_{th}.json';q=json.loads(p.read_text())
 assert q['source_sha256']==cohortrow['document_sha256'] and q['reference_sha256']==cohortrow['reference_sha256']
 assert q['output_sha256']==th and q['metric']==metricname
 return q['value'],p

def collect():
 frozen=json.loads((OUT/'frozen.json').read_text())
 for p,h in frozen['files'].items():assert sha(ROOT/p)==h,p
 cohort=[json.loads(s) for s in (OUT/'cohort/cohort.jsonl').read_text().splitlines()];assert len(cohort)==256
 models=json.loads((OUT/'calibration/weights.json').read_text())['models']
 rows=[];provenance={}
 for family in FAMILIES:
  for seed in SEEDS:
   group=[]
   for c in cohort:
    i=c['index'];p=OUT/f'generated/{family}/{seed}/{i:04}.json';m=json.loads(p.read_text())
    tpath=OUT/f'texts/{family}/{seed}/{i:04}.json';texts=json.loads(tpath.read_text())
    assert m['document_id']==c['id'] and m['freeze_sha256']==sha(OUT/'frozen.json')
    f=np.array([m['native_free_features'][k] for k in FEATURE_NAMES])
    risk=float(predict(f[None,:],models[family])[0]);assert np.isclose(risk,m['native_free_predicted_bertscore_gain'],rtol=1e-13,atol=1e-12)
    r=dict(family=family,generation_seed=seed,index=i,document_id=c['id'],document_sha256=c['document_sha256'],
      reference_sha256=c['reference_sha256'],**{k:float(v) for k,v in zip(FEATURE_NAMES,f)},
      risk_ridge=risk,risk_stopping_only=float(predict(f[None,5:8],models[family]['stopping_only'])[0]),
      risk_threefit_disagreement=m['native_free_secondary']['threefit_disagreement'],
      risk_negative_mean_log1p_gap=m['native_free_secondary']['negative_mean_log1p_gap'],
      state_sha256=m['states_sha256'],prefix_length=int(m['prefix_length']))
    assert np.isclose(r['risk_negative_mean_log1p_gap'],-f[0],rtol=0,atol=1e-12)
    for arm in ['native','autonomous17']:
     text=texts['texts'][arm];h=hashlib.sha256(text.encode()).hexdigest();assert h==texts['text_sha256'][arm]
     b,bp=metric(family,i,h,'bertscore',c);n,npth=metric(family,i,h,'nll',c)
     r.update({arm+'_bertscore':100.*b['f1'],arm+'_nll_eligible':int(n['eligible']),
       arm+'_nll':float(n['nll_sum']/n['tokens']) if n['eligible'] else 0.,arm+'_evaluator_tokens':int(n['tokens']),
       arm+'_empty':int(not text.strip()),arm+'_characters':len(text),
       arm+'_own_stop':int(texts['independent_boundaries'][arm]['end']),
       arm+'_terminating_eos':int(texts['independent_boundaries'][arm]['eos_present']),arm+'_text_sha256':h})
     for fp in [bp,npth]:provenance[str(fp.relative_to(ROOT))]=sha(fp)
    for key in ['sampling_with_native_and_capture_wall_seconds','native_terminal_forward_cuda_ms',
      'three_affine_heads_features_and_ridge_cuda_ms','three_affine_heads_features_and_ridge_wall_seconds',
      'isolated_affine17_features_and_ridge_cuda_ms','isolated_affine17_features_and_ridge_wall_seconds']:
     r[key]=float(m[key])
    provenance[str(p.relative_to(ROOT))]=sha(p);provenance[str(tpath.relative_to(ROOT))]=sha(tpath)
    group.append(r)
   for policy in POLICIES:
    for r,g in zip(group,choices(group,policy)):r['route_'+policy]=g
   rows+=group
 writecsv(OUT/'analysis/scalar_inputs.csv',rows)
 dump(OUT/'analysis/score_provenance.json',provenance)
 return rows

def interval(v):
 v=np.asarray(v,dtype=np.float64);assert v.ndim==1 and len(v)>0
 rng=np.random.default_rng(61032329);indices=rng.integers(0,len(v),size=(4000,len(v)))
 b=v[indices].mean(1)
 return dict(point=float(v.mean()),ci95=[float(t) for t in np.quantile(b,[.025,.975])],documents=len(v))

def toarray(rows,key):
 idx={(r['family'],int(r['generation_seed']),int(r['index'])):r for r in rows}
 return np.array([[[float(idx[(f,s,i)][key]) for s in SEEDS] for f in FAMILIES] for i in range(256)])

def summarize(rows):
 assert len(rows)==1024
 Bn=toarray(rows,'native_bertscore');Ba=toarray(rows,'autonomous17_bertscore')
 Nn=toarray(rows,'native_nll');Na=toarray(rows,'autonomous17_nll')
 eligible=(toarray(rows,'native_nll_eligible')>0)&(toarray(rows,'autonomous17_nll_eligible')>0)
 common=eligible.all((1,2));summary=dict(status='complete',prospective_frozen_test=True,
  documents=256,generations=1024,primary='ridge pooled BERTScore gain versus exact50%random expectation',
  inference='Fixed calibration weights and fixed selected decisions, document-paired4000bootstrap95%CI; no risk-fitting or fresh-batch-ranking uncertainty',
  nll_cohort=dict(documents=int(common.sum()),excluded_indices=np.flatnonzero(~common).tolist(),
   individual_ineligible=[{k:r[k] for k in ['family','generation_seed','index','document_id','native_nll_eligible','autonomous17_nll_eligible']} for r in rows if not(int(r['native_nll_eligible']) and int(r['autonomous17_nll_eligible']))],
   comparator_probability=.5,routing_decisions='Fixed from full256-document budget; eligible subset need not retain50%call rate'),
  policies={},stopping_ablation={},risk_diagnostics={},length_and_stop={},timing={})
 perdoc=[]
 for policy in POLICIES:
  G=toarray(rows,'route_'+policy)
  assert np.array_equal(G.sum(0),np.full((2,2),128))
  bd=(G-.5)*(Bn-Ba);nd=(G-.5)*(Na-Nn)
  pr=dict(bertscore={},nll={},actual_calls_per_family_seed=G.sum(0).astype(int).tolist(),descriptive_cost_proxy={})
  for name,fi in [('pooled',None),('elf',0),('aurora',1)]:
   b=bd.mean((1,2)) if fi is None else bd[:,fi,:].mean(1)
   pr['bertscore'][name]=interval(b)
   broute=(Ba+G*(Bn-Ba));bbaseline=.5*(Bn+Ba)
   pr['bertscore'][name].update(routed_score=float(broute.mean() if fi is None else broute[:,fi,:].mean()),
    random_expected_score=float(bbaseline.mean() if fi is None else bbaseline[:,fi,:].mean()),
    all_native_score=float(Bn.mean() if fi is None else Bn[:,fi,:].mean()),all_affine_score=float(Ba.mean() if fi is None else Ba[:,fi,:].mean()))
   if common.any():
    n=nd[common].mean((1,2)) if fi is None else nd[common,fi,:].mean(1)
    pr['nll'][name]=interval(n)
    nr=Na+G*(Nn-Na);rnd=.5*(Nn+Na)
    pr['nll'][name].update(routed_score=float(nr[common].mean() if fi is None else nr[common,fi,:].mean()),
     random_expected_score=float(rnd[common].mean() if fi is None else rnd[common,fi,:].mean()),
     actual_subset_native_call_fraction=float(G[common].mean() if fi is None else G[common,fi,:].mean()))
   else:pr['nll'][name]=dict(status='no_common_scorable_documents')
  summary['policies'][policy]=pr
  for family in FAMILIES:
   fr=[r for r in rows if r['family']==family]
   g=np.array([int(r['route_'+policy]) for r in fr]);ms=np.array([float(r['native_terminal_forward_cuda_ms']) for r in fr])
   prefix=np.array([int(r['prefix_length']) for r in fr])
   pr['descriptive_cost_proxy'][family]=dict(native_calls=int(g.sum()),outputs=len(fr),
    selected_native_forward_ms_per_output=float(np.mean(g*ms)),random_expected_native_forward_ms_per_output=float(.5*ms.mean()),
    selected_mean_prefix_length=float(prefix[g==1].mean()),unselected_mean_prefix_length=float(prefix[g==0].mean()),
    limitation='Same native call count is not same FLOPs/time; selected-native forward cost is a proxy from the all-outcome audit, not observed routed runtime or speedup.')
  for i in range(256):
   perdoc.append(dict(index=i,policy=policy,bertscore_gain=float(bd[i].mean()),
     elf_bertscore_gain=float(bd[i,0].mean()),aurora_bertscore_gain=float(bd[i,1].mean()),
     nll_common_eligible=int(common[i]),nll_gain=float(nd[i].mean()) if common[i] else 0.,
     elf_nll_gain=float(nd[i,0].mean()) if common[i] else 0.,aurora_nll_gain=float(nd[i,1].mean()) if common[i] else 0.))
 Gr=toarray(rows,'route_ridge');Gs=toarray(rows,'route_stopping_only')
 for name,fi in [('pooled',None),('elf',0),('aurora',1)]:
  v=(Gr-Gs)*(Bn-Ba);u=(Gr-Gs)*(Na-Nn)
  summary['stopping_ablation'][name]=dict(bertscore_full_minus_stopping=interval(v.mean((1,2)) if fi is None else v[:,fi,:].mean(1)))
  if common.any():summary['stopping_ablation'][name]['nll_benefit_full_minus_stopping']=interval(u[common].mean((1,2)) if fi is None else u[common,fi,:].mean(1))
 from scipy.stats import spearmanr
 for family in FAMILIES:
  fr=[r for r in rows if r['family']==family];y=np.array([float(r['native_bertscore'])-float(r['autonomous17_bertscore']) for r in fr])
  prediction=np.array([float(r['risk_ridge']) for r in fr]);rho=float(spearmanr(prediction,y).statistic)
  summary['risk_diagnostics'][family]=dict(mae=float(np.abs(prediction-y).mean()),signed_bias=float((prediction-y).mean()),spearman=rho if np.isfinite(rho) else None)
  summary['length_and_stop'][family]={}
  for arm in ['native','autonomous17']:
   summary['length_and_stop'][family][arm]={k:float(np.mean([float(r[arm+'_'+k]) for r in fr])) for k in ['empty','characters','own_stop','terminating_eos','evaluator_tokens']}
  summary['timing'][family]={k:dict(mean=float(np.mean([float(r[k]) for r in fr])),median=float(np.median([float(r[k]) for r in fr]))) for k in [
   'sampling_with_native_and_capture_wall_seconds','native_terminal_forward_cuda_ms','three_affine_heads_features_and_ridge_cuda_ms',
   'three_affine_heads_features_and_ridge_wall_seconds','isolated_affine17_features_and_ridge_cuda_ms','isolated_affine17_features_and_ridge_wall_seconds']}
 summary['timing']['limitation']='Descriptive shared-GPU instruments; all native and affine outputs computed. No measured end-to-end routed speedup. CUDA event intervals may include host synchronization gaps.'
 return summary,perdoc

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','recompute']);a=p.parse_args()
 rows=collect() if a.mode=='run' else readcsv(OUT/'analysis/scalar_inputs.csv')
 result,doc=summarize(rows);result['scalar_inputs_sha256']=sha(OUT/'analysis/scalar_inputs.csv')
 result['frozen_sha256']=sha(OUT/'frozen.json');writecsv(OUT/'analysis/per_document.csv',doc)
 dump(OUT/'analysis/summary.json',result)
 print(json.dumps(result['policies']['ridge'],indent=2),flush=True)

if __name__=='__main__':main()
