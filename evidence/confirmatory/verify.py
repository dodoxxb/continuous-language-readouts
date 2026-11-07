"""Offline verification of the new readout study (Python 3.10+ and NumPy).

Run from any directory: python path/to/confirmatory/verify.py
Recomputes statistics from sequence counts; does not reproduce GPU inference.
"""
from pathlib import Path
import collections,csv,hashlib,json,math,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader((ROOT/name).open()))
def check(a,b):assert abs(float(a)-float(b))<2e-11,(a,b)
def boot(c,d,other=None):
    c=np.mean(c,axis=0);d=np.asarray(d,dtype=float)
    if other is not None:c-=np.mean(other,axis=0)
    indices=np.random.default_rng(61013).integers(len(d),size=(4000,len(d)))
    values=c[indices].sum(1)/d[indices].sum(1)
    lo,hi=np.quantile(values,[.025,.975]);return c.sum()/d.sum(),lo,hi

protocol=json.loads((ROOT/'protocol.json').read_text());phash=hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest()
assert phash=='765e06c2701723826777df3f89277af197eec507461999b0ab42506f9bbdd522'
assert protocol['fit_seeds']==[17,29,43] and protocol['train_updates']==4096
fits=json.loads((ROOT/'fit_manifest.json').read_text());assert len(fits)==12
for fit in fits:
    assert fit['configuration']['protocol_sha256']==phash
    hist=fit['validation_history'];assert [r['step'] for r in hist]==[1024,2048,4096]
    best=max(hist,key=lambda r:r['validation']['agreement'])
    assert best['step']==fit['selected']['selected_update']
    check(best['validation']['agreement'],fit['selected']['validation_agreement'])
split=json.loads((ROOT/'elf_split.json').read_text())['sequence_ids']
assert [len(split[k]) for k in ['train','validation','test']]==[16384,512,1024]
for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(split[a])&set(split[b])
split=json.loads((ROOT/'langflow_split.json').read_text())
assert [len(split[k]) for k in ['train','validation','test']]==[1024,256,512]
for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not set(split[a])&set(split[b])

seq=read('per_sequence.csv');assert len(seq)==15360
groups=collections.defaultdict(list)
for row in seq:
    key=row['family'],int(row['train_sequences']),row['test_set'],int(row['fit_seed'])
    groups[key].append(row)
arrays={}
for key,rows in groups.items():
    rows.sort(key=lambda r:int(r['sequence_index']));assert [int(r['sequence_index']) for r in rows]==list(range(len(rows)))
    c=np.array([int(r['correct']) for r in rows]);d=np.array([int(r['valid_tokens']) for r in rows]);a=np.array([int(r['autonomous_correct']) for r in rows])
    assert np.all((0<=a)&(a<=c)&(c<=d)&(d>0));arrays[key]=(c,d,a)
for row in read('per_run.csv'):
    key=row['family'],int(row['train_sequences']),row['test_set'],int(row['fit_seed']);c,d,a=arrays[key]
    check(c.sum()/d.sum(),row['agreement']);check(np.mean(c/d),row['macro_agreement']);check(a.sum()/d.sum(),row['autonomous_eos_agreement']);assert len(d)==int(row['test_sequences']) and d.sum()==int(row['valid_tokens'])
for row in read('summary.csv'):
    keys=[(row['family'],int(row['train_sequences']),row['test_set'],s) for s in protocol['fit_seeds']]
    d=arrays[keys[0]][1];assert all(np.array_equal(arrays[k][1],d) for k in keys)
    c=np.stack([arrays[k][0] for k in keys]);m,lo,hi=boot(c,d)
    for val,key in [(m,'agreement_mean'),(lo,'sequence_ci_low'),(hi,'sequence_ci_high'),(np.std(c.sum(1)/d.sum(),ddof=1),'fit_seed_sd')]:check(val,row[key])
for row in read('paired_improvements.csv'):
    large=[arrays[('elf',16384,row['test_set'],s)][0] for s in protocol['fit_seeds']];small=[arrays[('elf',1024,row['test_set'],s)][0] for s in protocol['fit_seeds']];d=arrays[('elf',16384,row['test_set'],17)][1]
    assert all(np.array_equal(d,arrays[('elf',1024,row['test_set'],s)][1]) for s in protocol['fit_seeds'])
    for val,key in zip(boot(large,d,small),['delta','ci_low','ci_high']):check(val,row[key])
controls=collections.defaultdict(list)
for row in read('control_per_sequence.csv'):controls[row['family'],row['test_set'],int(row['step']),row['method']].append(row)
for row in read('lookup.csv')+read('langflow_phase.csv'):
    key=('elf',row['test_set'],-1,'lookup') if 'family' in row else ('langflow','test',int(row['step']),row['method'])
    rows=sorted(controls[key],key=lambda r:int(r['sequence_index']));assert [int(r['sequence_index']) for r in rows]==list(range(len(rows)))
    c=np.array([int(r['correct']) for r in rows]);d=np.array([int(r['valid_tokens']) for r in rows]);assert np.all((0<=c)&(c<=d))
    for val,name in zip(boot(c[None,:],d),['agreement','sequence_ci_low','sequence_ci_high']):check(val,row[name])
for file in ['tail_groups.csv','langflow_coverage.csv']:
    rows=read(file)
    for row in rows:
        c=int(row['correct']);d=int(row['tokens']);assert 0<=c<=d
        if d:check(c/d,row['agreement'])
    for seed in protocol['fit_seeds']:
        sub={r['group']:r for r in rows if int(r['fit_seed'])==seed};allrow=sub['all']
        for k in ['correct','tokens']:
            assert sum(int(sub[g][k]) for g in ['unseen_training_token','train_count_1_to_99','train_count_at_least_100'])==int(allrow[k])
            if file=='tail_groups.csv':
                assert int(sub['numeric'][k])+int(sub['non_numeric'][k])==int(allrow[k])
                assert sum(int(sub[g][k]) for g in ['margin_below_2','margin_2_to_8','margin_at_least_8'])==int(allrow[k])
            family='elf' if file=='tail_groups.csv' else 'langflow';c,d,_=arrays[(family,16384 if family=='elf' else 1024,'fresh_test' if family=='elf' else 'test',seed)]
            assert int(allrow[k])==int(c.sum() if k=='correct' else d.sum())
ppl=collections.defaultdict(list)
for row in read('ppl_per_sequence.csv'):ppl[row['method']].append(row)
assert len(ppl)==5 and all(len(rows)==512 for rows in ppl.values())
for row in read('ppl_summary.csv'):
    rows=ppl[row['method']];nll=sum(float(r['nll_sum']) for r in rows);tokens=sum(int(r['scored_tokens']) for r in rows)
    check(math.exp(nll/tokens),row['ppl']);assert tokens==int(row['scored_tokens']);assert sum(int(r['scored_tokens'])==0 for r in rows)==int(row['invalid_sequences'])==0
lineage=json.loads((ROOT/'native_checkpoint_check.json').read_text());correct=sum(r['correct'] for r in lineage['rows']);valid=sum(r['valid'] for r in lineage['rows']);assert valid-correct==13;check(correct/valid,lineage['native_agreement'])
print(json.dumps({'status':'passed','fits':12,'per_run_evaluations':21,'readout_sequence_rows':len(seq),'control_sequence_rows':sum(len(r) for r in controls.values()),'ppl_sequence_rows':2560,'bootstrap_draws':4000,'ci_scope':'conditional on fitted seeds and fixed split','protocol_sha256':phash,'historical_label_mismatches':13,'scope':'scalar recomputation; raw generation, training and inference require unbundled models and caches'},indent=2))
