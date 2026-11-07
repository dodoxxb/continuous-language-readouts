"""Paired, seed-stratified analysis; all prespecified signs/conditions retained."""
import argparse
import hashlib
import json
import math
import numpy as np
from aurora_study_20261022 import ROOT,OUT,FITS,POLICIES,FEATURES,protocol,verify_test_freeze,data_dir,csvread,csvwrite,features,predict,source_mask,dump,digest

def bootstrap(rows,key,condition='all',level=.95):
    p=protocol();conditions=list(p['test']['conditions']) if condition=='all' else [condition]
    seeds=p['test']['seeds'];n=p['test']['sequences_per_seed']
    rng=np.random.default_rng(61022501);samples=[];points=[]
    for seed in seeds:
        idx=rng.integers(n,size=(4000,n))
        for cond in conditions:
            values=np.full(n,np.nan)
            for row in rows:
                if row['generation_seed']==seed and row['condition']==cond:values[row['sequence']]=row[key]
            if np.isfinite(values).sum()<2:raise RuntimeError(f'Insufficient observations in {cond}/{seed}')
            points.append(np.nanmean(values));samples.append(np.nanmean(values[idx],axis=1))
    boot=np.mean(samples,axis=0);a=(1-level)/2
    return dict(mean=float(np.mean(points)),ci_low=float(np.quantile(boot,a)),ci_high=float(np.quantile(boot,1-a)),level=level)

def analyze(model_name='gpt2-large'):
    verify_test_freeze();p=protocol();fits=json.loads((OUT/'predictors.json').read_text())
    counts=np.load(OUT/'counts.npy');numeric=np.load(OUT/'numeric.npy')
    prediction_rows=[];sequence_rows=[];agreement=[];coverage=[];score_rows=[];validity=[];overlap=[]
    development=set()
    for split in ['training','validation','calibration']:
        y=np.load(OUT/'data'/split/'y.npy')
        development.update(hashlib.sha256(row.astype(np.int32).tobytes()).hexdigest() for row in y)
    seen=set()
    for cond in p['test']['conditions']:
        for seed in p['test']['seeds']:
            d=data_dir('test',cond,seed);y=np.load(d/'y.npy');margin=np.load(d/'margin.npy');v=source_mask(y)
            preds={f'linear{s}':np.load(d/f'linear{s}.npy') for s in FITS}
            repairs=np.load(d/'repairs.npz');arms={'native':y,**preds,**{f'repair_{k}':repairs[k] for k in POLICIES}}
            scores={a:csvread(d/f'scores_{model_name}_{a}.csv') for a in arms}
            assert all(len(r)==len(y) and [int(q['sequence']) for q in r]==list(range(len(y))) for r in scores.values())
            good=np.logical_and.reduce([np.array([int(q['eligible'])==1 for q in r]) for r in scores.values()])
            validity.append(dict(condition=cond,generation_seed=seed,total=len(y),common_scorable=int(good.sum()),excluded_indices=np.flatnonzero(~good).tolist()))
            per={a:np.array([float(q['nll_sum'])/int(q['tokens']) if int(q['tokens']) else np.nan for q in r]) for a,r in scores.items()}
            for a,ids in arms.items():
                correct=((ids==y)&v).sum(1);den=v.sum(1)
                assert np.array_equal(correct,[int(q['correct']) for q in scores[a]])
                if a.startswith('repair_'):
                    errors=((preds['linear17']!=y)&v).sum(1)
                    assert np.array_equal(den-correct,errors-errors//2)
                agreement.append(dict(condition=cond,generation_seed=seed,method=a,lexical_correct=int(correct.sum()),lexical_tokens=int(den.sum()),
                    lexical_agreement=float(correct.sum()/den.sum()),raw_all_position_agreement=float((ids==y).mean()),
                    native_eos_fraction=float((y==50256).mean()),predicted_eos_fraction=float((ids==50256).mean())))
                for i,q in enumerate(scores[a]):score_rows.append(dict(condition=cond,generation_seed=seed,common_scorable=int(good[i]),**q))
                if not a.startswith('linear'):continue
                groups={'unseen':counts[y]==0,'seen_1_99':(counts[y]>0)&(counts[y]<100),'frequent_100plus':counts[y]>=100,'numeric':numeric[y],'lowmargin':margin<2}
                for name,mask in groups.items():
                    m=mask&v;nn=int(m.sum());cc=int(((ids==y)&m).sum())
                    coverage.append(dict(condition=cond,generation_seed=seed,method=a,group=name,tokens=nn,correct=cc,agreement=cc/nn if nn else None,token_share=nn/int(v.sum())))
            prs=[]
            # The secondary scorer evaluates repairs, not a retrofitted primary predictor.
            if model_name=='gpt2-large':
                for fit_seed in FITS:
                    a=f'linear{fit_seed}';x=features(y,preds[a],margin,per['native']);target=per[a]-per['native']
                    base=predict(x[:,:3],fits[str(fit_seed)]['baseline']);tail=predict(x,fits[str(fit_seed)]['tail'])
                    xf=features(y,preds[a],margin,per['native'],rare_cutoff=fits[str(fit_seed)]['frequency_matched_count_cutoff'])
                    freq=predict(xf,fits[str(fit_seed)]['tail_frequency_matched'])
                    prs.append((target,base,tail,freq))
                    for i in np.flatnonzero(good):
                        prediction_rows.append(dict(condition=cond,generation_seed=seed,sequence=int(i),fit_seed=fit_seed,target=float(target[i]),
                            baseline_prediction=float(base[i]),tail_prediction=float(tail[i]),frequency_matched_prediction=float(freq[i]),
                            frequency_matched_rare_error_rate=float(xf[i,3]),**{k:float(x[i,j]) for j,k in enumerate(FEATURES)}))
            for i in np.flatnonzero(good):
                rand=float(np.mean([per[f'repair_{k}'][i] for k in POLICIES if k.startswith('random')]))
                row=dict(condition=cond,generation_seed=seed,sequence=int(i),
                    **{f'repair_{k}_gain':rand-float(per[f'repair_{k}'][i]) for k in ['rare','numeric','lowmargin']})
                if prs:
                    base_ae=np.mean([abs(t[i]-b[i]) for t,b,h,f in prs]);tail_ae=np.mean([abs(t[i]-h[i]) for t,b,h,f in prs])
                    freq_ae=np.mean([abs(t[i]-f[i]) for t,b,h,f in prs])
                    row.update(baseline_ae=float(base_ae),tail_ae=float(tail_ae),prediction_gain=float(base_ae-tail_ae),
                        baseline_se=float(np.mean([(t[i]-b[i])**2 for t,b,h,f in prs])),tail_se=float(np.mean([(t[i]-h[i])**2 for t,b,h,f in prs])),
                        frequency_matched_ae=float(freq_ae),frequency_matched_gain=float(base_ae-freq_ae))
                sequence_rows.append(row)
            sig=[hashlib.sha256(row.astype(np.int32).tobytes()).hexdigest() for row in y]
            overlap.append(dict(condition=cond,generation_seed=seed,development_exact_overlaps=sum(s in development for s in sig),within_pool_duplicates=len(sig)-len(set(sig)),earlier_test_pool_overlaps=sum(s in seen for s in sig)))
            seen.update(sig)
    prediction_summary=[];repair_summary=[];ppl=[]
    for cond in ['all']+list(p['test']['conditions']):
        rows=[r for r in sequence_rows if cond=='all' or r['condition']==cond]
        if model_name=='gpt2-large':
            stat=bootstrap(sequence_rows,'prediction_gain',cond,.975 if cond=='all' else .95)
            bm=bootstrap(sequence_rows,'baseline_ae',cond)['mean'];tm=bootstrap(sequence_rows,'tail_ae',cond)['mean']
            fs=bootstrap(sequence_rows,'frequency_matched_gain',cond,.95)
            prediction_summary.append(dict(condition=cond,sequences=len(rows),baseline_mae=bm,tail_mae=tm,relative_mae_reduction=(bm-tm)/bm,**stat,
                frequency_matched_mae=bm-fs['mean'],frequency_matched_gain=fs['mean'],frequency_matched_ci_low=fs['ci_low'],frequency_matched_ci_high=fs['ci_high']))
        for policy in ['rare','numeric','lowmargin']:
            level=.975 if model_name=='gpt2-large' and cond=='all' and policy=='rare' else .95
            repair_summary.append(dict(condition=cond,policy=policy,sequences=len(rows),**bootstrap(sequence_rows,f'repair_{policy}_gain',cond,level)))
        for arm in ['native']+[f'linear{s}' for s in FITS]+[f'repair_{k}' for k in POLICIES]:
            rows=[r for r in score_rows if r['method']==arm and int(r['common_scorable']) and (cond=='all' or r['condition']==cond)]
            total_nll=sum(float(r['nll_sum']) for r in rows);tokens=sum(int(r['tokens']) for r in rows)
            ppl.append(dict(condition=cond,method=arm,scored_tokens=tokens,nll=total_nll/tokens,ppl=math.exp(total_nll/tokens)))
    dest=OUT/'analysis'/model_name;dest.mkdir(parents=True,exist_ok=True)
    for name,rows in [('predictions',prediction_rows),('per_sequence',sequence_rows),('agreement',agreement),('coverage',coverage),('scores',score_rows),('prediction_summary',prediction_summary),('repair_summary',repair_summary),('ppl',ppl),('overlap',overlap)]:
        if rows:csvwrite(dest/(name+'.csv'),rows)
    dump(dest/'summary.json',dict(evaluator=model_name,protocol_sha256=digest(OUT/'protocol.json'),prediction=prediction_summary,repair=repair_summary,validity=validity,overlap=overlap))
    print(json.dumps(dict(prediction=prediction_summary,repair=repair_summary),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',default='gpt2-large');a=p.parse_args();analyze(a.model)
