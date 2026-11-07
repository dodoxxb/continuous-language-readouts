"""Paired-document analysis of all fixed task outcomes."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/task_readout_20261022'
FAMILIES=['elf','aurora'];SEEDS=[61022801,61024801]
ARMS=['native','linear17','linear29','linear43','repair_random11','repair_random23','repair_random37','repair_rare','repair_numeric','repair_lowmargin','autonomous17']
METRICS=['rouge1','rouge2','rougeL','bertscore','summac','nll']
N=1024

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def key(i,text):return f'{i:04}_'+hashlib.sha256(text.encode()).hexdigest()
def ci(x,draws,level=.95):
    # x = document x paired replicas; NaN only in predeclared eligibility sets.
    a=np.isfinite(x);num=np.where(a,x,0).sum(axis=1);den=a.sum(axis=1)
    total=den.sum()
    if not total:return dict(mean=None,lower=None,upper=None,level=level,documents=0,generations=0)
    bn=num[draws].sum(1);bd=den[draws].sum(1);vals=bn[bd>0]/bd[bd>0]
    lo,hi=np.quantile(vals,[(1-level)/2,(1+level)/2])
    return dict(mean=float(num.sum()/total),lower=float(lo),upper=float(hi),level=level,documents=int((den>0).sum()),generations=int(total))

def main():
    dest=OUT/'analysis';dest.mkdir(exist_ok=True)
    draws=np.random.default_rng(61022901).integers(0,N,size=(4000,N),dtype=np.int32)
    np.savez_compressed(dest/'bootstrap_indices.npz',indices=draws)
    values=np.full((len(METRICS),N,2,2,len(ARMS)),np.nan);editable=np.zeros((N,2,2),bool)
    coverage=[];agreements=[];flat=[]
    for fi,family in enumerate(FAMILIES):
        for si,seed in enumerate(SEEDS):
            cov=dict(family=family,seed=seed,documents=N,editable=0,empty_native=0,native_eos_absent=0,autonomous_early=0,autonomous_late=0,
                     nll_common_eligible=0,nll_autonomous_eligible=0,source_truncated_summac=0,empty_chunks_summac=0,
                     native_tokens=0,errors=0,repairs=0,canvas_min=10000,canvas_max=0)
            correct={a:0 for a in ARMS if a!='autonomous17'};total=0
            for i in range(N):
                q=json.loads((OUT/f'texts/{family}/{seed}/{i:04}.json').read_text())
                editable[i,fi,si]=q['editable'];cov['editable']+=int(q['editable']);cov['native_tokens']+=q['valid_tokens'];cov['errors']+=q['errors'];cov['repairs']+=q['repair_budget']
                cov['empty_native']+=not bool(q['texts']['native']);cov['native_eos_absent']+=not q['native_eos_present']
                cov['autonomous_early']+=q['readout_ends']['autonomous17']<q['native_end'];cov['autonomous_late']+=q['readout_ends']['autonomous17']>q['native_end']
                cov['canvas_min']=min(cov['canvas_min'],q['canvas_length']);cov['canvas_max']=max(cov['canvas_max'],q['canvas_length'])
                total+=q['valid_tokens']
                for a in correct:correct[a]+=q['agreement'][a]['correct']
                for ai,arm in enumerate(ARMS):
                    k=key(i,q['texts'][arm]);scores={name:json.loads((OUT/f'scores/{name}/{family}/{k}.json').read_text())['value'] for name in ['rouge','bertscore','summac','nll']}
                    vals=[scores['rouge'][m]['f1'] for m in METRICS[:3]]+[scores['bertscore']['f1'],scores['summac']['score'],scores['nll']['nll_sum']/scores['nll']['tokens'] if scores['nll']['eligible'] else np.nan]
                    values[:,i,fi,si,ai]=vals
                    flat.append(dict(family=family,seed=seed,index=i,arm=arm,editable=q['editable'],**{m:float(v) if np.isfinite(v) else '' for m,v in zip(METRICS,vals)}))
                    if arm=='native':
                        cov['source_truncated_summac']+=scores['summac']['source_truncated'];cov['empty_chunks_summac']+=scores['summac']['empty_chunks']
                common=np.isfinite(values[-1,i,fi,si,:10]).all()
                cov['nll_common_eligible']+=int(common);cov['nll_autonomous_eligible']+=int(np.isfinite(values[-1,i,fi,si,10]))
                if not common:values[-1,i,fi,si,:10]=np.nan
            coverage.append(cov)
            agreements.extend(dict(family=family,seed=seed,arm=a,correct=c,total=total,agreement=c/total if total else 0.) for a,c in correct.items())
    assert np.isfinite(values[:-1]).all()
    np.savez_compressed(dest/'analysis_arrays.npz',values=values,editable=editable)
    summary=[];effects=[]
    for group in ['pooled']+FAMILIES:
        fidx=slice(None) if group=='pooled' else slice(FAMILIES.index(group),FAMILIES.index(group)+1)
        for mi,m in enumerate(METRICS):
            current=values[mi,:,fidx,:,:].copy()
            if group=='pooled' and m=='nll':
                # Equal four-replica weights: pooled NLL uses documents scorable
                # in every main arm of both families and generation seeds.
                eligible=np.isfinite(current[:,:,:,:10]).all(axis=(1,2,3))
                current[~eligible,:,:,:10]=np.nan
            for ai,arm in enumerate(ARMS):
                x=current[:,:,:,ai].reshape(N,-1)
                summary.append(dict(group=group,metric=m,arm=arm,**ci(x,draws)))
            random=current[:,:,:,4:7].mean(-1)
            for policy,ai in [('rare',7),('numeric',8),('lowmargin',9)]:
                # Positive always favors targeted repair; NLL uses lower better.
                x=(current[:,:,:,ai]-random)*( -1 if m=='nll' else 1)
                primary=group=='pooled' and policy=='rare' and m in ['bertscore','summac']
                effects.append(dict(group=group,metric=m,policy=policy,cohort='all',primary=primary,**ci(x.reshape(N,-1),draws,.975 if primary else .95)))
                xe=np.where(editable[:,fidx,:],x,np.nan)
                effects.append(dict(group=group,metric=m,policy=policy,cohort='editable',primary=False,**ci(xe.reshape(N,-1),draws)))
    out=dict(families=FAMILIES,generation_seeds=SEEDS,documents=N,generations=N*4,
        primary=[x for x in effects if x['primary']],effects=effects,scores=summary,coverage=coverage,agreement=agreements,
        interpretation='Positive targeted-minus-random effect means improved proxy score; NLL sign reversed. Native oracle is not truth. Intervals resample documents across paired seeds and families.')
    dump(dest/'summary.json',out)
    with (dest/'per_generation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(dict(primary=out['primary'],coverage=coverage),indent=2))

if __name__=='__main__':main()
