"""Analyze the frozen, post-initial-result LangFlow token-budget extension."""
from pathlib import Path
import json,statistics
import numpy as np
from confirmatory_readout_20261013 import OUT,P,dump
from confirmatory_analyze_20261013 import csvwrite,interval
dest=OUT/'followup';dest.mkdir(exist_ok=True)
y=np.load(OUT/'data/langflow/test/y.npy');special=json.loads((OUT/'langflow_special_token_audit.json').read_text())['special_ids'];normal=~np.isin(y,special)
seq=[];runs=[];summary=[];coverage=[];allcorrect={};den=np.full(len(y),y.shape[1]);nonden=normal.sum(1)
for budget,family in [(1024,'langflow'),(8192,'langflow_extended')]:
    counts=np.bincount(np.load(OUT/'data'/family/'train_flat/y.npy'),minlength=30522)
    allcorrect[budget]=[];allnon=[]
    for seed in P['fit_seeds']:
        folder=OUT/'fits'/family/f'n{budget}_seed{seed}';pack=np.load(folder/'test.npz');selected=json.loads((folder/'selected.json').read_text())
        pred=pack['pred'];c=(pred==y).sum(1);cn=((pred==y)&normal).sum(1)
        assert np.array_equal(c,pack['correct']) and np.array_equal(den,pack['denominator']);assert abs(c.sum()/den.sum()-selected['test']['agreement'])<1e-12
        allcorrect[budget].append(c);allnon.append(cn)
        runs.append({'train_sequences':budget,'train_tokens':budget*128,'fit_seed':seed,'selected_update':selected['selected_update'],'validation_agreement':selected['validation_agreement'],'agreement':c.sum()/den.sum(),'nonspecial_agreement':cn.sum()/nonden.sum()})
        for i in range(len(y)):seq.append({'train_sequences':budget,'fit_seed':seed,'sequence_index':i,'correct':int(c[i]),'valid_tokens':int(den[i]),'nonspecial_correct':int(cn[i]),'nonspecial_tokens':int(nonden[i])})
        for group,m in [('all',np.ones_like(y,dtype=bool)),('unseen_training_token',counts[y]==0),('train_count_1_to_99',(counts[y]>0)&(counts[y]<100)),('train_count_at_least_100',counts[y]>=100)]:
            c0=int(((pred==y)&m).sum());d0=int(m.sum());coverage.append({'train_sequences':budget,'fit_seed':seed,'group':group,'correct':c0,'tokens':d0,'agreement':c0/d0 if d0 else None})
    m,lo,hi=interval(allcorrect[budget],den);nm,nlo,nhi=interval(allnon,nonden)
    summary.append({'train_sequences':budget,'train_tokens':budget*128,'fit_seeds':3,'agreement_mean':m,'fit_seed_sd':statistics.stdev(float(c.sum()/den.sum()) for c in allcorrect[budget]),'sequence_ci_low':lo,'sequence_ci_high':hi,'nonspecial_agreement_mean':nm,'nonspecial_sequence_ci_low':nlo,'nonspecial_sequence_ci_high':nhi,'native_special_positions':int(y.size-nonden.sum()),'native_positions':int(y.size)})
m,lo,hi=interval(allcorrect[8192],den,other=allcorrect[1024]);paired=[{'comparison':'8192 minus 1024 LangFlow training sequences','delta':m,'ci_low':lo,'ci_high':hi}]
for file,rows in [('per_sequence.csv',seq),('per_run.csv',runs),('summary.csv',summary),('coverage.csv',coverage),('paired_improvement.csv',paired)]:csvwrite(dest/file,rows)
dump(dest/'analysis.json',{'timing':'Post-initial-result token-budget follow-up; protocol frozen before additional generation and fits','summary':summary,'paired':paired,'scope':'Same fixed LangFlow test, approximately matched available training-token count to smallest ELF pool. No architecture ranking.'})
print(json.dumps({'summary':summary,'paired':paired},indent=2))
