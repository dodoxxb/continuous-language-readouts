"""Offline numerical audit of retained outcomes, not a rerun of model inference."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

def check(root):
    checks=[]
    def equal(a,b):assert np.allclose(a,b,atol=1e-12,rtol=1e-12),(a,b)
    c=root/'calm';ll=root/'llaco'
    summary=json.loads((c/'summary.json').read_text())
    boot=np.random.default_rng(61102009).integers(512,size=(4000,512))
    for r in summary:
        d=np.load(c/'test'/(r['condition']+'.npz'));y=d['native']
        keys=['17','29','43'] if r['fit']=='three_fit_mean' else [r['fit']]
        a=np.stack([(d[k]==y).reshape(512,-1).mean(1) for k in keys]).mean(0)
        b=np.stack([(d[k]==y).all(-1).reshape(512,-1).mean(1) for k in keys]).mean(0)
        equal(a.mean(),r['token_agreement']);equal(b.mean(),r['chunk_agreement'])
        equal(np.quantile(a[boot].mean(1),[.025,.975]),r['token_ci'])
        equal(np.quantile(b[boot].mean(1),[.025,.975]),r['chunk_ci'])
        equal((y==d['reference']).mean(),r['native_reference'])
        equal((y==d['native_base']).mean(),r['native_radial_preservation'])
    checks.append('CALM primary: every condition, fit, document interval and native-reference result')
    old=[json.loads(l)['document_sha256'] for l in (c/'cohort/documents.jsonl').read_text().splitlines()]
    fresh=[json.loads(l)['document_sha256'] for l in (c/'fit_control/fresh_documents.jsonl').read_text().splitlines()]
    assert len(old)==len(set(old))==4864 and len(fresh)==len(set(fresh))==512 and not set(old)&set(fresh)
    f=json.loads((c/'fit_control/summary.json').read_text());d=np.load(c/'fit_control/fresh_predictions.npz');y=d['native']
    boot=np.random.default_rng(61102209).integers(512,size=(4000,512));base=(d['ridge_17']==y).reshape(512,-1).mean(1)
    for r in f['results']:
        ss=[17] if r['family']=='ridge' else [17,29,43]
        a=np.stack([(d[f'{r["family"]}_{s}']==y).reshape(512,-1).mean(1) for s in ss]).mean(0);delta=a-base
        equal(a.mean(),r['agreement']);equal(np.quantile(a[boot].mean(1),[.025,.975]),r['ci'])
        equal(delta.mean(),r['gain_over_ridge']);equal(np.quantile(delta[boot].mean(1),[.025,.975]),r['gain_ci'])
    checks.append('CALM follow-up: fresh hash exclusions, all policies, paired gains and intervals')
    g=json.loads((c/'generated_summary.json').read_text());d=np.load(c/'generated_predictions.npz');y=d['native']
    assert y.shape==(512*63,4)
    boot=np.random.default_rng(61102022).integers(512,size=(4000,512))
    for r in g['results']:
        keys=['17','29','43'] if r['fit']=='three_fit_mean' else ['tied17','tied29','tied43'] if r['fit']=='tied_three_fit_mean' else [r['fit']]
        a=np.stack([(d[k]==y).reshape(512,-1).mean(1) for k in keys]).mean(0)
        equal(a.mean(),r['agreement']);equal(np.quantile(a[boot].mean(1),[.025,.975]),r['ci'])
        if 'gain_ci' in r:
            delta=a-(d['ridge']==y).reshape(512,-1).mean(1)
            equal(delta.mean(),r['gain_over_ridge']);equal(np.quantile(delta[boot].mean(1),[.025,.975]),r['gain_ci'])
    checks.append('CALM conditional one-step prior: all native/readout outcomes and transfer intervals')
    g=json.loads((c/'generated_fit/summary.json').read_text());d=np.load(c/'generated_fit/test_predictions.npz');y=d['native']
    assert y.shape==(512*63,4) and g['documents']==512 and g['chunks']==512*63
    boot=np.random.default_rng(61102309).integers(512,size=(4000,512));means={}
    for r in g['results']:
        a=np.stack([(d[f'{r["family"]}_{s}']==y).reshape(512,-1).mean(1) for s in [17,29,43]]).mean(0)
        means[r['family']]=a;equal(a.mean(),r['agreement']);equal(np.quantile(a[boot].mean(1),[.025,.975]),r['ci'])
        for s in [17,29,43]:
            equal((d[f'{r["family"]}_{s}']==y).mean(),r['individual_agreements'][str(s)])
            folder=c/'generated_fit'/f'{r["family"]}_seed{s}'
            h=json.loads((folder/'history.json').read_text());m=json.loads((folder/'manifest.json').read_text())
            assert m['selected_step']==max(h,key=lambda row:row['agreement'])['step']
            equal(m['validation_agreement'],max(row['agreement'] for row in h))
            coef=folder/'selected_coef.pt'
            if coef.exists():assert hashlib.sha256(coef.read_bytes()).hexdigest()==m['coef_sha256']
    delta=means['prior']-means['posterior'];equal(delta.mean(),g['gain']);equal(np.quantile(delta[boot].mean(1),[.025,.975]),g['gain_ci'])
    checks.append('CALM matched fitting populations: all six fits, validation-only selections, paired fresh-prior gains and intervals')
    l=json.loads((ll/'summary.json').read_text());rows=list(csv.DictReader((ll/'per_source.csv').open()))
    for r in l['geometry']:
        rr=[v for v in rows if v['endpoint']==r['endpoint'] and float(v['gamma'])==r['gamma'] and v['split']=='test'];assert len(rr)==768
        for k in ['cosine_correct','euclidean_correct','adapter_correct','adapter_cosine_disagree','source_rank','posterior_max']:equal(np.mean([float(v[k]) for v in rr]),r[k])
    ss=list(csv.DictReader((ll/'sparse_per_source.csv').open()))
    for r in ss:
        assert float(r['l2_error'])<=float(r['bound'])+1e-4
        if int(r['sparse']):assert -1e-6<=float(r['omitted_mass'])<=float(r['epsilon'])+1e-6 and int(r['kept_k'])<=64
        else:assert int(r['kept_k'])==131072
    for r in l['sparse']:
        rr=[v for v in ss if v['split']=='test' and float(v['epsilon'])==r['epsilon'] and float(v['gamma'])==r['gamma']]
        assert len(rr)==768;equal(np.mean([int(v['sparse']) for v in rr]),r['sparse_fraction'])
    for r in l['runtime']:equal(np.median(r['measurements']),r['median_ms']);assert len(r['measurements'])==20
    checks.append('LLaCO: paired vocabulary outcomes, mass-budget bounds, all fallback rates and runtime medians')
    return dict(status='passed',checks=checks,scope='Offline arithmetic and retained-prediction audit. Does not rerun pretrained inference or verify unbundled weights.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();j=check(a.root)
    if a.output:a.output.write_text(json.dumps(j,indent=2)+'\n')
    print(json.dumps(j,indent=2))
