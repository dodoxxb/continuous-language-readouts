"""Portable NumPy-only verification of predictive and repair scalar evidence.

Run beside the exported CSV/JSON files. This reproduces aggregation and inference,
not original GPU measurements, generation or evaluator forward passes.
"""
from pathlib import Path
import csv,json,math
import numpy as np
P=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader((P/name).open()))
def eq(a,b):assert abs(float(a)-float(b))<1e-10,(a,b)
def ranks(x):
    order=np.argsort(x,kind='stable');result=np.empty(len(x));i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        result[order[i:j]]=(i+j-1)/2.+1.;i=j
    return result
def boot(values,strata,level):
    rng=np.random.default_rng(61013203);groups=[np.flatnonzero(strata==k) for k in np.unique(strata)];out=np.zeros(4000);shared={}
    for g in groups:
        key=(str(strata[g[0]]).rsplit('_',1)[-1],len(g))
        if key not in shared:shared[key]=rng.integers(len(g),size=(4000,len(g)))
        out+=values[g][shared[key]].mean(1)/len(groups)
    alpha=(1-level)/2
    return np.mean([values[g].mean() for g in groups]),*np.quantile(out,[alpha,1-alpha])
def run():
    protocol=json.loads((P/'protocol.json').read_text());fits=json.loads((P/'predictors.json').read_text());cal=read('calibration.csv');pred=read('predictions.csv');seq=read('per_sequence.csv');scores=read('text_scores.csv')
    cols=protocol['tail_features'];lookup={}
    for seed in protocol['fit_seeds']:
        rr=[r for r in cal if int(r['fit_seed'])==seed];x=np.array([[float(r[k]) for k in cols] for r in rr]);y=np.array([float(r['target']) for r in rr])
        for name,n in [('baseline',3),('tail',6)]:
            xx=x[:,:n];mu=xx.mean(0);sd=xx.std(0);sd=np.where(sd>1e-12,sd,1.);z=np.c_[np.ones(len(xx)),(xx-mu)/sd];reg=np.eye(n+1)*10.;reg[0,0]=0.;beta=np.linalg.solve(z.T@z+reg,z.T@y);f=fits[str(seed)][name]
            assert np.allclose(mu,f['mean'],atol=1e-12) and np.allclose(sd,f['std'],atol=1e-12) and np.allclose(beta,f['beta'],atol=1e-10)
    for r in pred:
        f=fits[r['fit_seed']];x=np.array([float(r[k]) for k in cols])
        for name,n in [('baseline',3),('tail',6)]:
            est=np.r_[1.,(x[:n]-f[name]['mean'])/f[name]['std']]@f[name]['beta'];eq(est,r[name+'_prediction'])
        key=(r['condition'],r['generation_seed'],r['sequence']);lookup.setdefault(key,[]).append(r)
    textlookup={}
    for r in scores:textlookup.setdefault((r['condition'],r['generation_seed'],r['sequence']),{})[r['method']]=r
    for r in seq:
        key=(r['condition'],r['generation_seed'],r['sequence']);rr=lookup[key];ss=textlookup[key]
        yy=np.array([float(z['target']) for z in rr]);bp=np.array([float(z['baseline_prediction']) for z in rr]);tp=np.array([float(z['tail_prediction']) for z in rr]);eq(np.abs(yy-bp).mean(),r['baseline_ae']);eq(np.abs(yy-tp).mean(),r['tail_ae'])
        native=float(ss['native']['nll_sum'])/int(ss['native']['scored_tokens'])
        for z in rr:
            target=ss['linear_seed'+z['fit_seed']];eq(float(target['nll_sum'])/int(target['scored_tokens'])-native,z['target'])
        original=ss['linear_seed17'];errors=int(original['valid_tokens'])-int(original['correct']);random=[]
        for name,z in ss.items():
            if name.startswith('repair_'):
                assert int(z['valid_tokens'])-int(z['correct'])==errors-errors//2
                if name.startswith('repair_random'):random.append(float(z['nll_sum'])/int(z['scored_tokens']))
        for policy in ['rare','numeric','lowmargin']:
            z=ss['repair_'+policy];eq(np.mean(random)-float(z['nll_sum'])/int(z['scored_tokens']),r['repair_'+policy+'_gain'])
    for r in read('prediction_summary.csv'):
        rr=[z for z in seq if r['condition']=='all' or z['condition']==r['condition']];base=np.array([float(z['baseline_ae']) for z in rr]);tail=np.array([float(z['tail_ae']) for z in rr]);strata=np.array([z['condition']+'_'+z['generation_seed'] for z in rr]);eq(base.mean(),r['baseline_mae']);eq(tail.mean(),r['tail_mae'])
        for x,k in zip(boot(base-tail,strata,float(r['interval_level'])),['mae_reduction','ci_low','ci_high']):eq(x,r[k])
        ps=[z for z in pred if r['condition']=='all' or z['condition']==r['condition']];yy=np.array([float(z['target']) for z in ps]);variance=((yy-yy.mean())**2).sum()
        for method in ['baseline','tail']:
            estimate=np.array([float(z[method+'_prediction']) for z in ps]);squared=(estimate-yy)**2
            eq(np.sqrt(squared.mean()),r[method+'_rmse']);eq(1-squared.sum()/variance,r[method+'_r2']);eq(np.corrcoef(ranks(yy),ranks(estimate))[0,1],r[method+'_spearman'])
    for r in read('repair_summary.csv'):
        rr=[z for z in seq if r['condition']=='all' or z['condition']==r['condition']];val=np.array([float(z['repair_'+r['policy']+'_gain']) for z in rr]);strata=np.array([z['condition']+'_'+z['generation_seed'] for z in rr])
        for x,k in zip(boot(val,strata,float(r['interval_level'])),['mean_random_minus_target_nll','ci_low','ci_high']):eq(x,r[k])
    for r in read('ppl_summary.csv'):
        rr=[z for z in scores if z['condition']==r['condition'] and z['method']==r['method']];eq(math.exp(sum(float(z['nll_sum']) for z in rr)/sum(int(z['scored_tokens']) for z in rr)),r['ppl']);eq(sum(int(z['correct']) for z in rr)/sum(int(z['valid_tokens']) for z in rr),r['agreement'])
    secondary_count=0
    if (P/'secondary_text_scores.csv').exists():
        secondary=read('secondary_text_scores.csv');secondary_seq=read('secondary_per_sequence.csv');secondary_count=len(secondary);sl={}
        for r in secondary:
            key=(r['condition'],r['generation_seed'],r['sequence']);sl.setdefault(key,{})[r['method']]=r
            old=textlookup[key][r['method']];assert r['correct']==old['correct'] and r['valid_tokens']==old['valid_tokens']
        for r in secondary_seq:
            ss=sl[(r['condition'],r['generation_seed'],r['sequence'])];random=[float(z['nll_sum'])/int(z['scored_tokens']) for name,z in ss.items() if name.startswith('repair_random')]
            for policy in ['rare','numeric','lowmargin']:
                z=ss['repair_'+policy];eq(np.mean(random)-float(z['nll_sum'])/int(z['scored_tokens']),r['repair_'+policy+'_gain'])
        for r in read('secondary_repair_summary.csv'):
            rr=[z for z in secondary_seq if r['condition']=='all' or z['condition']==r['condition']];val=np.array([float(z['repair_'+r['policy']+'_gain']) for z in rr]);strata=np.array([z['condition']+'_'+z['generation_seed'] for z in rr])
            for x,k in zip(boot(val,strata,.95),['mean_random_minus_target_nll','ci_low','ci_high']):eq(x,r[k])
        for r in read('secondary_ppl_summary.csv'):
            rr=[z for z in secondary if z['condition']==r['condition'] and z['method']==r['method']];eq(math.exp(sum(float(z['nll_sum']) for z in rr)/sum(int(z['scored_tokens']) for z in rr)),r['ppl'])
    print(json.dumps(dict(status='passed',calibration_rows=len(cal),prediction_rows=len(pred),sequence_clusters=len(seq),text_score_rows=len(scores),secondary_text_score_rows=secondary_count,checks=['fixed ridge fits','frozen predictions','native-NLL targets','exact matched repair counts','paired bootstrap','corpus PPL','secondary evaluator']),indent=2))
if __name__=='__main__':run()
