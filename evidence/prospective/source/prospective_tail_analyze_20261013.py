"""Analyze the protocol-frozen test; retain unsuccessful predictions."""
import json, math
import numpy as np
from scipy.stats import spearmanr
from prospective_tail_20261013 import OUT,OLD,CONDITIONS,FITS,POLICIES,TAIL,features,predict,readcsv,writecsv,dump,digest

def bootstrap(values,strata,level=.95):
    rng=np.random.default_rng(61013203);groups=[np.flatnonzero(strata==k) for k in np.unique(strata)]
    boot=np.zeros(4000)
    # Reuse indices across checkpoint conditions sharing a generation seed.
    # This retains common-random-number dependence rather than treating those
    # trajectories as independent. All prespecified test pools have equal size.
    shared={}
    for g in groups:
        key=(str(strata[g[0]]).rsplit('_',1)[-1],len(g))
        if key not in shared:shared[key]=rng.integers(len(g),size=(4000,len(g)))
        idx=shared[key]
        boot+=values[g][idx].mean(1)/len(groups)
    a=(1-level)/2
    return float(np.mean([values[g].mean() for g in groups])),float(np.quantile(boot,a)),float(np.quantile(boot,1-a))

def analyze():
    p=json.loads((OUT/'protocol.json').read_text());models=json.loads((OUT/'predictors.json').read_text())
    predictions=[];sequence=[];texts=[];validity=[];agreement=[];allpacks=[]
    for cond in CONDITIONS:
        for gseed in p['generation_seeds']:
            dest=OUT/'test'/cond/str(gseed);pack=np.load(dest/'tokens.npz');repair=np.load(dest/'repairs.npz')
            y=pack['native'];v=pack['valid'];n=len(y);allpacks.append((cond,gseed,y,v))
            armnames=['native']+[f'linear_seed{s}' for s in FITS]+POLICIES
            rr={a:readcsv(dest/f'ppl_{a}.csv') for a in armnames}
            ns={a:np.array([int(r['scored_tokens']) for r in rr[a]]) for a in armnames}
            nlls={a:np.array([float(r['nll_sum']) for r in rr[a]]) for a in armnames}
            good=np.logical_and.reduce([ns[a]>0 for a in armnames])
            validity.append(dict(condition=cond,generation_seed=gseed,sequences=n,common_scorable=int(good.sum()),excluded_indices=np.flatnonzero(~good).tolist()))
            per={a:np.divide(nlls[a],ns[a],out=np.zeros(n),where=ns[a]>0) for a in armnames}
            for a in armnames:
                ids=y if a=='native' else pack[a] if a.startswith('linear') else repair[a]
                correct=((ids==y)&v).sum(1);den=v.sum(1)
                assert np.array_equal(correct,[int(r['correct']) for r in rr[a]])
                if a in POLICIES:
                    errors=((pack['linear_seed17']!=y)&v).sum(1)
                    assert np.array_equal(den-correct,errors-errors//2)
                kept=(ids==1).cumsum(1)==0;autonomous=((np.where(kept,ids,0)==y)&v).sum(1)
                agreement.append(dict(condition=cond,generation_seed=gseed,method=a,correct=int(correct.sum()),valid_tokens=int(den.sum()),agreement=float(correct.sum()/den.sum()),autonomous_agreement=float(autonomous.sum()/den.sum())))
                for i in range(n):texts.append(dict(condition=cond,generation_seed=gseed,sequence=i,method=a,nll_sum=float(nlls[a][i]),scored_tokens=int(ns[a][i]),correct=int(correct[i]),valid_tokens=int(den[i]),common_scorable=int(good[i])))
            pred_arrays={}
            for fit in FITS:
                a=f'linear_seed{fit}';x=features(y,v,pack[a],pack['margin'],per['native']);target=per[a]-per['native']
                base=predict(x[:,:3],models[str(fit)]['baseline']);tail=predict(x,models[str(fit)]['tail'])
                pred_arrays[fit]=(target,base,tail)
                for i in range(n):
                    predictions.append(dict(condition=cond,generation_seed=gseed,sequence=i,fit_seed=fit,common_scorable=int(good[i]),target=float(target[i]),baseline_prediction=float(base[i]),tail_prediction=float(tail[i]),**{k:float(x[i,j]) for j,k in enumerate(TAIL)}))
            for i in np.flatnonzero(good):
                targets=np.array([pred_arrays[f][0][i] for f in FITS]);bp=np.array([pred_arrays[f][1][i] for f in FITS]);tp=np.array([pred_arrays[f][2][i] for f in FITS])
                rand=np.mean([per[a][i] for a in POLICIES if a.startswith('repair_random')])
                sequence.append(dict(condition=cond,generation_seed=gseed,sequence=int(i),baseline_ae=float(np.abs(targets-bp).mean()),tail_ae=float(np.abs(targets-tp).mean()),baseline_se=float(((targets-bp)**2).mean()),tail_se=float(((targets-tp)**2).mean()),repair_rare_gain=float(rand-per['repair_rare'][i]),repair_numeric_gain=float(rand-per['repair_numeric'][i]),repair_lowmargin_gain=float(rand-per['repair_lowmargin'][i])))
    summary=[];interventions=[]
    for cond in ['all']+list(CONDITIONS):
        rows=[r for r in sequence if cond=='all' or r['condition']==cond]
        strata=np.array([f"{r['condition']}_{r['generation_seed']}" for r in rows]);base=np.array([r['baseline_ae'] for r in rows]);tail=np.array([r['tail_ae'] for r in rows])
        mean,lo,hi=bootstrap(base-tail,strata,.975 if cond=='all' else .95)
        prs=[r for r in predictions if r['common_scorable'] and (cond=='all' or r['condition']==cond)]
        yy=np.array([r['target'] for r in prs]);bb=np.array([r['baseline_prediction'] for r in prs]);tt=np.array([r['tail_prediction'] for r in prs]);var=((yy-yy.mean())**2).sum()
        summary.append(dict(condition=cond,sequences=len(rows),baseline_mae=float(base.mean()),tail_mae=float(tail.mean()),mae_reduction=mean,ci_low=lo,ci_high=hi,interval_level=.975 if cond=='all' else .95,baseline_rmse=float(np.sqrt(np.mean([r['baseline_se'] for r in rows]))),tail_rmse=float(np.sqrt(np.mean([r['tail_se'] for r in rows]))),baseline_r2=float(1-((yy-bb)**2).sum()/var),tail_r2=float(1-((yy-tt)**2).sum()/var),baseline_spearman=float(spearmanr(yy,bb).statistic),tail_spearman=float(spearmanr(yy,tt).statistic)))
        for policy in ['rare','numeric','lowmargin']:
            val=np.array([r[f'repair_{policy}_gain'] for r in rows]);level=.975 if cond=='all' and policy=='rare' else .95
            mean,lo,hi=bootstrap(val,strata,level)
            interventions.append(dict(condition=cond,policy=policy,mean_random_minus_target_nll=mean,ci_low=lo,ci_high=hi,interval_level=level,sequences=len(rows)))
    ppl=[]
    for cond in list(CONDITIONS):
        for a in ['native']+[f'linear_seed{s}' for s in FITS]+POLICIES:
            rr=[r for r in texts if r['condition']==cond and r['method']==a]
            sums=sum(r['nll_sum'] for r in rr);nt=sum(r['scored_tokens'] for r in rr);den=sum(r['valid_tokens'] for r in rr);cc=sum(r['correct'] for r in rr)
            ppl.append(dict(condition=cond,method=a,ppl=math.exp(sums/nt),agreement=cc/den,scored_tokens=nt,valid_tokens=den,correct=cc))
    # Exact native-sequence overlap against old calibration/test and historical training.
    def signatures(y,v):return [__import__('hashlib').sha256(row[mask].astype(np.int64).tobytes()).hexdigest() for row,mask in zip(y,v)]
    oldsig=set()
    for name in ['validation','test','fresh_test']:
        d=OLD/'data/elf'/name;oldsig.update(signatures(np.load(d/'y.npy'),np.load(d/'valid.npy')))
    flat=np.load(OLD/'data/elf/train/y.npy',mmap_mode='r');off=np.load(OLD/'data/elf/train/offsets.npy')
    import hashlib
    oldsig.update(hashlib.sha256(np.asarray(flat[off[i]:off[i+1]],dtype=np.int64).tobytes()).hexdigest() for i in range(len(off)-1))
    seen=set();overlap=[]
    for cond,gseed,y,v in allpacks:
        sig=signatures(y,v);overlap.append(dict(condition=cond,generation_seed=gseed,old_exact_overlaps=sum(s in oldsig for s in sig),within_pool_duplicates=len(sig)-len(set(sig)),earlier_new_pool_overlaps=sum(s in seen for s in sig)));seen.update(sig)
    for name,rows in [('predictions',predictions),('per_sequence',sequence),('text_scores',texts),('agreement',agreement),('prediction_summary',summary),('repair_summary',interventions),('ppl_summary',ppl),('overlap',overlap)]:writecsv(OUT/(name+'.csv'),rows)
    dump(OUT/'analysis.json',dict(protocol_sha256=digest(OUT/'protocol.json'),prediction=summary,repair=interventions,validity=validity,overlap=overlap,primary_prediction_supported=summary[0]['ci_low']>0,primary_rare_repair_supported=interventions[0]['ci_low']>0))
    print(json.dumps(dict(prediction=summary,repair=interventions,validity=validity),indent=2))

if __name__=='__main__':analyze()
