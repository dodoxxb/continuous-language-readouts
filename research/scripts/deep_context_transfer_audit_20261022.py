"""CPU-only diagnostic of frozen AURORA adapter transfer and feature scaling.

No refit, model selection, new deployed model, or full-test ablation occurs.
The axis decomposition is an explanatory intervention selected after observing
transfer failure. It must not be reported as a prospective quality improvement.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
CONTEXT=ROOT/'outputs/deep_mechanism_20261022/context'
OUT=CONTEXT/'transfer_audit'
FAMILY=CONTEXT/'aurora'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def quantile(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(zip(['min','p01','p05','median','p95','p99','max'],
                    map(float,np.quantile(x,[0,.01,.05,.5,.95,.99,1]))))
def context_rows(x,pos):
    out=np.zeros((len(pos),x.shape[-1]),np.float32)
    for lag in range(1,5):
        ok=pos>=lag;out[ok]+=np.asarray(x[pos[ok]-lag],np.float32)
    return (out/np.maximum(np.minimum(pos,4),1)[:,None]).astype(np.float32)


def collect():
    groups={};provenance=[]
    for split in ['train','validation']:
        path=FAMILY/(split+'.npz');z=np.load(path)
        ix=np.linspace(0,len(z['y'])-1,128).round().astype(int)
        groups[split]=dict(x=z['x'][ix],context=z['context'][ix],
                          shuffled=z['context'][z['donor'][ix]],y=z['y'][ix],expected={})
        provenance.append(dict(scope=split,path=str(path.relative_to(ROOT)),indices=ix.tolist()))
    collected={scope:{k:[] for k in ['x','context','shuffled','y','expected']}
               for scope in ['owt','xsum']}
    for d in sorted((ROOT/'outputs/aurora_replication_20261022/data/test').glob('*/*')):
        x=np.load(d/'x.npy',mmap_mode='r');y=np.load(d/'y.npy')
        expected=np.load(FAMILY/'evaluation/owt'/f'{d.parent.name}_{d.name}.npz')
        for i in [0,64,128,192]:
            eligible=np.flatnonzero(y[i]!=50256)
            pos=eligible[np.linspace(0,len(eligible)-1,4).round().astype(int)]
            q=collected['owt'];q['x'].append(x[i,pos]);q['context'].append(context_rows(x[i],pos))
            q['shuffled'].append(context_rows(x[(i+len(x)//2)%len(x)],pos));q['y'].append(y[i,pos])
            q['expected'].append({k:expected[k][i,pos] for k in expected.files})
            provenance.append(dict(scope='owt',directory=str(d.relative_to(ROOT)),sequence=i,positions=pos.tolist()))
    task=ROOT/'outputs/task_readout_20261022'
    for d in sorted((task/'generated/aurora').iterdir()):
        paths=sorted(d.glob('[0-9][0-9][0-9][0-9].npz'))
        for i in range(0,len(paths),128):
            path=paths[i];z=np.load(path);meta=json.loads(path.with_suffix('.json').read_text())
            mask=np.load(task/'texts/aurora'/d.name/path.name)['valid'];eligible=np.flatnonzero(mask)
            if not len(eligible):continue
            positions=eligible[np.linspace(0,len(eligible)-1,min(4,len(eligible))).round().astype(int)]
            fullpos=positions+meta['prefix_length'];otherpath=paths[(i+len(paths)//2)%len(paths)]
            other=np.load(otherpath);om=json.loads(otherpath.with_suffix('.json').read_text())
            donorpos=np.minimum(positions+om['prefix_length'],len(other['full_x'])-1)
            q=collected['xsum'];q['x'].append(z['x'][positions]);q['y'].append(z['native'][positions])
            q['context'].append(context_rows(z['full_x'],fullpos))
            q['shuffled'].append(context_rows(other['full_x'],donorpos))
            expected=np.load(FAMILY/'evaluation/xsum'/d.name/path.name)
            q['expected'].append({k:expected[k][positions] for k in expected.files})
            provenance.append(dict(scope='xsum',path=str(path.relative_to(ROOT)),positions=positions.tolist(),
                                   donor=str(otherpath.relative_to(ROOT))))
    for scope,q in collected.items():
        groups[scope]={k:np.concatenate(v) for k,v in q.items() if k!='expected'}
        groups[scope]['expected']={key:np.concatenate([r[key] for r in q['expected']])
                                  for key in q['expected'][0]}
    return groups,provenance


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    basepath=ROOT/'outputs/aurora_replication_20261022/fits/seed17/selected.pt'
    base=torch.load(basepath,map_location='cpu',weights_only=True)
    stats=json.loads((FAMILY/'normalization.json').read_text());groups,provenance=collect()
    models={p.parent.name:torch.load(p,map_location='cpu',weights_only=True)
            for p in sorted((FAMILY/'fits').glob('*/selected.pt'))}
    normalizers={};summary={};per_token=[]
    for feature in ['focal','context']:
        ss=np.asarray(stats[feature]['std']);normalizers[feature]=dict(
            std_quantiles=quantile(ss),floor_axes=np.flatnonzero(ss==1e-4).tolist(),
            smallest_axis=int(np.argmin(ss)),smallest_axis_mean=stats[feature]['mean'][int(np.argmin(ss))])
    assert normalizers['focal']['floor_axes']==normalizers['context']['floor_axes']==[86]
    axis=86
    for key,state in models.items():
        st=stats['focal' if key.startswith('focal') else 'context']
        assert np.allclose(state['mean'].numpy(),st['mean'])
        assert np.allclose(state['std'].numpy(),st['std'])
        assert all(torch.isfinite(t).all() for t in state.values())
    for scope,g in groups.items():
        result=dict(tokens=len(g['y']),features={},models={})
        for feature in ['focal','context','shuffled']:
            q=g['x' if feature=='focal' else feature].astype(np.float64)
            st=stats['focal' if feature=='focal' else 'context']
            z=(q-np.asarray(st['mean']))/np.asarray(st['std'])
            result['features'][feature]=dict(raw_rms=quantile(np.sqrt((q*q).mean(-1))),
                standardized_rms=quantile(np.sqrt((z*z).mean(-1))),raw_axis86=quantile(q[:,axis]),
                standardized_axis86=quantile(z[:,axis]),
                axis86_standardized_energy_fraction=quantile(z[:,axis]**2/(z*z).sum(-1)))
        parts={key:[] for key in models};basepred=[];baserms=[]
        for start in range(0,len(g['y']),16):
            end=min(start+16,len(g['y']));x=torch.from_numpy(g['x'][start:end].copy())
            logits=F.linear(x,base['weight'],base['bias']);bp=logits.argmax(-1);basepred.extend(bp.tolist())
            baserms.extend(torch.sqrt((logits*logits).mean(-1)).tolist())
            for key,state in models.items():
                feature=key.split('_')[0];q=torch.from_numpy(g['x' if feature=='focal' else feature][start:end].copy())
                z=(q-state['mean'])/state['std']
                delta=F.linear(F.linear(z,state['v.weight']),state['u.weight'],state['u.bias'])
                coefficient=state['u.weight']@state['v.weight'][:,axis]
                axis_delta=z[:,axis,None]*coefficient[None,:]
                remainder=delta-axis_delta
                full=(logits+delta).argmax(-1);without=(logits+remainder).argmax(-1)
                direct=z.clone();direct[:,axis]=0
                direct_delta=F.linear(F.linear(direct,state['v.weight']),state['u.weight'],state['u.bias'])
                maxerr=float((direct_delta-remainder).abs().max())
                # Cancellation in very large deltas creates normal FP32 error.
                assert maxerr<2e-3,(scope,key,maxerr)
                for j in range(end-start):
                    expected=None if key not in g['expected'] else int(g['expected'][key][start+j])
                    item=dict(scope=scope,arm=key,index=start+j,native=int(g['y'][start+j]),
                        base=int(bp[j]),full=int(full[j]),diagnostic_axis86_zero=int(without[j]),
                        expected=expected,delta_rms=float(torch.sqrt((delta[j]**2).mean())),
                        axis86_delta_rms=float(torch.sqrt((axis_delta[j]**2).mean())),
                        remainder_delta_rms=float(torch.sqrt((remainder[j]**2).mean())),
                        decomposition_max_absolute_error=maxerr)
                    parts[key].append(item);per_token.append(item)
        result['base']=dict(agreement=float(np.mean(np.asarray(basepred)==g['y'])),logit_rms=quantile(baserms),
            cached_prediction_mismatches=None if 'base' not in g['expected'] else
            int(np.sum(np.asarray(basepred)!=g['expected']['base'])))
        for key,rr in parts.items():
            result['models'][key]=dict(
                full_agreement=float(np.mean([r['full']==r['native'] for r in rr])),
                diagnostic_axis86_zero_agreement=float(np.mean([r['diagnostic_axis86_zero']==r['native'] for r in rr])),
                diagnostic_axis86_zero_matches_base=float(np.mean([r['diagnostic_axis86_zero']==r['base'] for r in rr])),
                full_cached_prediction_mismatches=None if rr[0]['expected'] is None else sum(r['full']!=r['expected'] for r in rr),
                delta_rms=quantile([r['delta_rms'] for r in rr]),
                axis86_delta_rms=quantile([r['axis86_delta_rms'] for r in rr]),
                remainder_delta_rms=quantile([r['remainder_delta_rms'] for r in rr]),
                decomposition_max_absolute_error=max(r['decomposition_max_absolute_error'] for r in rr))
        summary[scope]=result
        print(scope,'tokens',len(g['y']),'base',result['base']['agreement'],flush=True)
    report=dict(scope='Post-hoc CPU-only numerical diagnosis on deterministic representative positions; no retraining or deployment',
        axis_selection='Only coordinate whose original OWT normalization std was floored to1e-4; identified after observed XSum collapse',
        warning='The axis86-zero calculation is a selected explanatory ablation, not a new validated model or a reported task-quality improvement.',
        normalizers=normalizers,summary=summary,positions=provenance,
        source_sha256=dict(normalization=sha(FAMILY/'normalization.json'),base=sha(basepath),
            **{key:sha(FAMILY/'fits'/key/'selected.pt') for key in models}),
        all_model_states_finite=True,normalizer_state_dict_matches_json=True)
    dump(OUT/'aurora_transfer_numerical_audit.json',report)
    dump(OUT/'aurora_representative_token_diagnostics.json',per_token)
    print('Saved',OUT,flush=True)


if __name__=='__main__':main()
