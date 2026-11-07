"""Independent checks of native labels, exact edits, scoring and frozen predictors."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/aurora_replication_20261022'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def close(a,b):assert np.allclose(a,b,atol=1e-9,rtol=1e-8),(a,b)

def main():
    p=json.loads((OUT/'protocol.json').read_text());fr=json.loads((OUT/'test_frozen.json').read_text())
    assert sha(OUT/'protocol.json')==json.loads((OUT/'protocol_frozen.json').read_text())['sha256']
    for path,h in fr['files'].items():assert sha(ROOT/path)==h,path
    for path,h in p['upstream_files'].items():assert sha(ROOT/path)==h,path
    for info in p['weights']:assert sha(ROOT/info['path'])==info['sha256'],info['path']
    for path,h in json.loads((OUT/'source/model_assets.json').read_text())['files'].items():assert sha(ROOT/path)==h,path
    counts=np.load(OUT/'counts.npy');numeric=np.load(OUT/'numeric.npy');fits=json.loads((OUT/'predictors.json').read_text())
    training=np.load(OUT/'data/training/y.npy');assert np.array_equal(counts,np.bincount(training[training!=50256],minlength=50257))
    report=[];indexed={}
    for cond in p['test']['conditions']:
        for seed in p['test']['seeds']:
            d=OUT/'data/test'/cond/str(seed);y=np.load(d/'y.npy');v=y!=50256;margin=np.load(d/'margin.npy')
            ready=json.loads((d/'ready.json').read_text())
            for name,h in ready['files'].items():assert sha(d/name)==h,str(d/name)
            assert y.shape==(p['test']['sequences_per_seed'],p['sampler']['length'])
            pred=np.load(d/'linear17.npy');wrong=(pred!=y)&v;n=wrong.sum(1)//2
            repairs=np.load(d/'repairs.npz');assert len(repairs.files)==6
            for policy in repairs.files:
                ids=repairs[policy];changed=ids!=pred
                assert not (changed&~wrong).any(),(cond,seed,policy,'non-error edited')
                assert np.array_equal(changed.sum(1),n),(cond,seed,policy,'wrong count')
                assert np.array_equal(ids[changed],y[changed]),(cond,seed,policy,'non-native correction')
                if policy in ['rare','numeric','lowmargin']:
                    priority=counts[y] if policy=='rare' else -numeric[y].astype(int) if policy=='numeric' else margin
                    for i in range(len(y)):
                        chosen=priority[i,changed[i]];remaining=priority[i,wrong[i]&~changed[i]]
                        if len(chosen) and len(remaining):assert chosen.max()<=remaining.min()
            for model in ['gpt2-large','opt-125m']:
                for fp in sorted(d.glob(f'scores_{model}_*.csv')):
                    rr=rows(fp);assert len(rr)==len(y)
                    assert [int(q['sequence']) for q in rr]==list(range(len(y)))
                    method=rr[0]['method']
                    ids=y if method=='native' else np.load(d/(method+'.npy')) if method.startswith('linear') else repairs[method[7:]]
                    assert np.array_equal(((ids==y)&v).sum(1),[int(q['correct']) for q in rr])
                    for i,q in enumerate(rr):
                        eligible=2<=int(q['uncapped_tokens'])<=1024
                        assert bool(int(q['eligible']))==eligible
                        if eligible:assert int(q['tokens'])==int(q['uncapped_tokens'])-1 and np.isfinite(float(q['nll_sum']))
                        else:assert int(q['tokens'])==0
                        if model=='gpt2-large':indexed[(cond,seed,i,method)]=q
                assert len(list(d.glob(f'scores_{model}_*.csv')))==10
            report.append(dict(condition=cond,generation_seed=seed,sequences=len(y),source_errors=int(wrong.sum()),corrections_per_arm=int(n.sum()),lexical_positions=int(v.sum())))
    predrows=rows(OUT/'analysis/gpt2-large/predictions.csv')
    for q in predrows:
        key=(q['condition'],int(q['generation_seed']),int(q['sequence']));fit_seed=q['fit_seed']
        native=indexed[(*key,'native')];linear=indexed[(*key,'linear'+fit_seed)]
        actual=float(linear['nll_sum'])/int(linear['tokens'])-float(native['nll_sum'])/int(native['tokens'])
        close(float(q['target']),actual)
        names=p['regression']['features'];x=np.array([float(q[k]) for k in names])
        for name,n in [('baseline',3),('tail',6)]:
            ff=fits[fit_seed][name];prediction=np.r_[1.,(x[:n]-np.array(ff['mean']))/np.array(ff['std'])]@np.array(ff['beta'])
            close(float(q[name+'_prediction']),prediction)
        xf=x.copy();xf[3]=float(q['frequency_matched_rare_error_rate']);ff=fits[fit_seed]['tail_frequency_matched']
        prediction=np.r_[1.,(xf-np.array(ff['mean']))/np.array(ff['std'])]@np.array(ff['beta'])
        close(float(q['frequency_matched_prediction']),prediction)
    result=dict(status='passed',frozen_files_and_upstream_hashes_verified=True,weight_hashes_verified=True,
        exact_equal_count_native_repairs_verified=True,priority_order_verified=True,scorer_token_boundaries_verified=True,
        independent_prediction_reconstruction_verified=True,pools=report,primary_prediction_records=len(predrows))
    temporary=OUT/'integrity.json.tmp'
    temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(OUT/'integrity.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
