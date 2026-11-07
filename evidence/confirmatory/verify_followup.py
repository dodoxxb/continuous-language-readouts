"""Portable scalar checks for the post-initial-result LangFlow extension."""
from pathlib import Path
import csv,hashlib,json,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader((ROOT/'followup'/name).open()))
def close(a,b):assert abs(float(a)-float(b))<2e-11,(a,b)
def boot(c,d,other=None):
    c=np.mean(c,axis=0);d=np.asarray(d,dtype=float)
    if other is not None:c-=np.mean(other,axis=0)
    indices=np.random.default_rng(61013).integers(len(d),size=(4000,len(d)));b=c[indices].sum(1)/d[indices].sum(1)
    return c.sum()/d.sum(),*np.quantile(b,[.025,.975])
hash0=hashlib.sha256((ROOT/'langflow_followup_protocol.json').read_bytes()).hexdigest();assert hash0=='4ce232902a61ad86ea014a612ddbd6e7efe1c44f7d1135453738954c2000dc58'
fits=json.loads((ROOT/'followup_fit_manifest.json').read_text());assert len(fits)==3
for fit in fits:
    assert fit['followup_configuration']['followup_protocol_sha256']==hash0
    best=max(fit['validation_history'],key=lambda x:x['validation']['agreement']);assert best['step']==fit['selected']['selected_update'];close(best['validation']['agreement'],fit['selected']['validation_agreement'])
rows=read('per_sequence.csv');assert len(rows)==3072
arrays={}
for budget in [1024,8192]:
    for seed in [17,29,43]:
        rr=sorted([r for r in rows if int(r['train_sequences'])==budget and int(r['fit_seed'])==seed],key=lambda r:int(r['sequence_index']))
        assert [int(r['sequence_index']) for r in rr]==list(range(512))
        c,d,nc,nd=[np.array([int(r[k]) for r in rr]) for k in ['correct','valid_tokens','nonspecial_correct','nonspecial_tokens']]
        assert np.all((nc<=c)&(c<=d)&(nc<=nd)&(nd<=d)&(nc>=0)&(nd>0));arrays[budget,seed]=c,d,nc,nd
initial=list(csv.DictReader((ROOT/'per_sequence.csv').open()))
for seed in [17,29,43]:
    rr=sorted([r for r in initial if r['family']=='langflow' and int(r['fit_seed'])==seed],key=lambda r:int(r['sequence_index']))
    assert np.array_equal(arrays[1024,seed][0],np.array([int(r['correct']) for r in rr]))
for row in read('per_run.csv'):
    c,d,nc,nd=arrays[int(row['train_sequences']),int(row['fit_seed'])];close(c.sum()/d.sum(),row['agreement']);close(nc.sum()/nd.sum(),row['nonspecial_agreement'])
for row in read('summary.csv'):
    budget=int(row['train_sequences']);cc=[arrays[budget,s][0] for s in [17,29,43]];nn=[arrays[budget,s][2] for s in [17,29,43]];d=arrays[budget,17][1];nd=arrays[budget,17][3]
    for s in [17,29,43]:assert np.array_equal(arrays[budget,s][1],d) and np.array_equal(arrays[budget,s][3],nd)
    for v,k in zip(boot(cc,d),['agreement_mean','sequence_ci_low','sequence_ci_high']):close(v,row[k])
    for v,k in zip(boot(nn,nd),['nonspecial_agreement_mean','nonspecial_sequence_ci_low','nonspecial_sequence_ci_high']):close(v,row[k])
    close(statistics.stdev(float(c.sum()/d.sum()) for c in cc),row['fit_seed_sd']);assert int(row['native_special_positions'])==int(d.sum()-nd.sum())==3172
assert np.array_equal(arrays[1024,17][1],arrays[8192,17][1])
row=read('paired_improvement.csv')[0]
for v,k in zip(boot([arrays[8192,s][0] for s in [17,29,43]],arrays[8192,17][1],[arrays[1024,s][0] for s in [17,29,43]]),['delta','ci_low','ci_high']):close(v,row[k])
coverage=read('coverage.csv')
for row in coverage:
    c=int(row['correct']);d=int(row['tokens']);assert 0<=c<=d
    if d:close(c/d,row['agreement'])
for budget in [1024,8192]:
    for seed in [17,29,43]:
        rr={r['group']:r for r in coverage if int(r['train_sequences'])==budget and int(r['fit_seed'])==seed}
        for key,index in [('correct',0),('tokens',1)]:assert sum(int(rr[g][key]) for g in ['unseen_training_token','train_count_1_to_99','train_count_at_least_100'])==int(rr['all'][key])==int(arrays[budget,seed][index].sum())
print(json.dumps({'status':'passed','new_fits':3,'same_test_reused':True,'paired_budget_delta_recomputed':True,'special_token_sensitivity_recomputed':True,'timing':'post-initial-result follow-up','followup_protocol_sha256':hash0},indent=2))
