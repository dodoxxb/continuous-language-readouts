"""Post-protocol regression controls; only saved calibration/test features.

No original predictor or test outcome is changed. --verify recomputes without
writing. NumPy plus the standard library suffice for portable execution.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

FEATURES=['error_rate','log_length','native_nll','rare_error_rate','numeric_error_rate','lowmargin_error_rate']
METHODS=['original_baseline','original_tail','duplicate_all','duplicate_error','ols_baseline','ols_tail']

def read_csv(path):
    with path.open() as handle:return list(csv.DictReader(handle))

def design(x):
    mean=x.mean(0);std=x.std(0);std=np.where(std>1e-12,std,1.)
    return np.column_stack([np.ones(len(x)),(x-mean)/std]),mean,std

def fit(x,target,penalty=None):
    z,mean,std=design(x)
    if penalty is None:
        beta,_,rank,singular=np.linalg.lstsq(z,target,rcond=1e-10)
    else:
        alpha=np.full(x.shape[1],float(penalty)) if np.ndim(penalty)==0 else np.asarray(penalty)
        beta=np.linalg.solve(z.T@z+np.diag(np.r_[0.,alpha]),z.T@target)
        singular=np.linalg.svd(z,compute_uv=False);rank=np.linalg.matrix_rank(z)
    return dict(mean=mean.tolist(),std=std.tolist(),beta=beta.tolist(),rank=int(rank),
                condition_number=float(singular[0]/singular[-1]) if singular[-1]>0 else None)

def predict(x,model):
    return np.column_stack([np.ones(len(x)),(x-model['mean'])/model['std']])@np.array(model['beta'])

def bootstrap(records,conditions,seeds,n,condition,key):
    rng=np.random.default_rng(61022501);points=[];draws=[]
    for seed in seeds:
        indices=rng.integers(n,size=(4000,n))
        for cond in conditions if condition=='all' else [condition]:
            values=np.full(n,np.nan)
            for q in records:
                if q['condition']==cond and q['generation_seed']==seed:values[q['sequence']]=q[key]
            assert np.isfinite(values).sum()>=2
            points.append(np.nanmean(values));draws.append(np.nanmean(values[indices],axis=1))
    sample=np.mean(draws,axis=0)
    return dict(mean=float(np.mean(points)),ci_low=float(np.quantile(sample,.025)),ci_high=float(np.quantile(sample,.975)))

def compute(root,family):
    protocol=json.loads((root/'protocol.json').read_text())
    if family=='aurora':
        test=protocol['test'];testpath=root/'analysis/gpt2-large/predictions.csv'
        conditions=list(test['conditions']);seeds=test['seeds'];n=test['sequences_per_seed']
    else:
        testpath=root/'predictions.csv';conditions=list(protocol['conditions']);seeds=protocol['generation_seeds'];n=protocol['sequences_per_seed']
    calibration=read_csv(root/'calibration.csv');testrows=read_csv(testpath)
    groups={};models={}
    for seed in [17,29,43]:
        cal=[q for q in calibration if int(q['fit_seed'])==seed]
        rr=[q for q in testrows if int(q['fit_seed'])==seed and q.get('common_scorable','1')=='1']
        x=np.array([[float(q[k]) for k in FEATURES] for q in cal]);target=np.array([float(q['target']) for q in cal])
        xt=np.array([[float(q[k]) for k in FEATURES] for q in rr]);truth=np.array([float(q['target']) for q in rr])
        transforms={
            'duplicate_all':(np.tile(x[:,:3],(1,2)),np.tile(xt[:,:3],(1,2)),10.),
            'duplicate_error':(np.column_stack([x[:,:3],x[:,0],x[:,0],x[:,0]]),np.column_stack([xt[:,:3],xt[:,0],xt[:,0],xt[:,0]]),10.),
            'ols_baseline':(x[:,:3],xt[:,:3],None),'ols_tail':(x,xt,None)}
        fitted={name:fit(train,target,penalty) for name,(train,test,penalty) in transforms.items()}
        predicted={name:predict(test,fitted[name]) for name,(train,test,penalty) in transforms.items()}
        # Algebraic controls: duplication changes penalties but not the function class.
        assert np.allclose(predicted['duplicate_all'],predict(xt[:,:3],fit(x[:,:3],target,5.)),atol=1e-10,rtol=1e-9)
        assert np.allclose(predicted['duplicate_error'],predict(xt[:,:3],fit(x[:,:3],target,[2.5,10.,10.])),atol=1e-10,rtol=1e-9)
        models[str(seed)]=fitted
        predicted.update(original_baseline=np.array([float(q['baseline_prediction']) for q in rr]),
                         original_tail=np.array([float(q['tail_prediction']) for q in rr]))
        for i,q in enumerate(rr):
            key=(q['condition'],int(q['generation_seed']),int(q['sequence']))
            groups.setdefault(key,[]).append({k:abs(float(predicted[k][i])-float(truth[i])) for k in METHODS})
    records=[]
    for (condition,seed,sequence),items in sorted(groups.items()):
        assert len(items)==3
        q=dict(condition=condition,generation_seed=seed,sequence=sequence,
               **{k:float(np.mean([item[k] for item in items])) for k in METHODS})
        q.update(original_ridge_gain=q['original_baseline']-q['original_tail'],
                 tail_over_duplicate_all=q['duplicate_all']-q['original_tail'],
                 tail_over_duplicate_error=q['duplicate_error']-q['original_tail'],
                 ols_tail_gain=q['ols_baseline']-q['ols_tail'])
        records.append(q)
    summary=[]
    for cond in ['all']+conditions:
        maes={k:bootstrap(records,conditions,seeds,n,cond,k)['mean'] for k in METHODS}
        contrasts={k:bootstrap(records,conditions,seeds,n,cond,k) for k in
                   ['original_ridge_gain','tail_over_duplicate_all','tail_over_duplicate_error','ols_tail_gain']}
        summary.append(dict(condition=cond,sequences=sum(cond=='all' or q['condition']==cond for q in records),mae=maes,contrasts=contrasts))
    return dict(family=family,scope='Post-protocol descriptive robustness audit; original primary results unchanged',
                confidence_level=.95,bootstrap_draws=4000,bootstrap_seed=61022501,
                duplicate_penalty_equivalence_verified=True,summary=summary,models=models,
                input_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'protocol.json',root/'calibration.csv',testpath]}),records

def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert np.isclose(a,b,atol=1e-10,rtol=1e-9),(a,b)
    else:assert a==b,(a,b)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--family',choices=['aurora','elf'],required=True);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args();result,records=compute(args.input_root,args.family)
    if args.verify:
        compare(result,json.loads((args.output_dir/'summary.json').read_text()))
        print(args.family+' regression robustness recomputation passed');return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'per_sequence.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    temporary=args.output_dir/'summary.json.tmp'
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(args.output_dir/'summary.json')
    print(json.dumps({'family':args.family,'pooled':result['summary'][0]},indent=2))

if __name__=='__main__':main()
