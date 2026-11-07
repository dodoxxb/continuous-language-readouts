"""Post hoc all-policy equal-evaluator-edit-dose sensitivity; CPU tokenization."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import hashlib,json
import numpy as np
from transformers import AutoTokenizer
from deep_repair_confound_20261022 import bootstrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/deep_mechanism_20261022/repair'
SRC=ROOT/'outputs/task_readout_20261022'
FAMILIES=['elf','aurora'];SEEDS=[61022801,61024801]
POLICIES=['random11','random23','random37','rare','numeric','lowmargin']
protocol=json.loads((OUT/'equal_evaluator_dose_extension_protocol.json').read_text())
assert hashlib.sha256((OUT/'arrays.npz').read_bytes()).hexdigest()==protocol['original_arrays_sha256']
assert hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest()==protocol['original_summary_sha256']
tok=AutoTokenizer.from_pretrained(ROOT/'models/gpt2-large',local_files_only=True)
doses=np.full((1024,2,2,6),-1,np.int32)
lengths=np.zeros((1024,2,2,7),np.int32)
for fi,family in enumerate(FAMILIES):
    for si,seed in enumerate(SEEDS):
        for i in range(1024):
            q=json.loads((SRC/f'texts/{family}/{seed}/{i:04}.json').read_text())
            b=np.array(tok.encode(q['texts']['linear17'],add_special_tokens=False))
            lengths[i,fi,si,0]=len(b)
            for pi,policy in enumerate(POLICIES):
                r=np.array(tok.encode(q['texts']['repair_'+policy],add_special_tokens=False))
                lengths[i,fi,si,pi+1]=len(r)
                if len(b)==len(r):doses[i,fi,si,pi]=np.count_nonzero(b!=r)
        print('tokenized',family,seed,flush=True)
z=np.load(OUT/'arrays.npz');fixed=z['fixed_position'];budgets=z['budgets']
v=np.load(SRC/'analysis/analysis_arrays.npz')['values']
draws=np.load(SRC/'analysis/bootstrap_indices.npz')['indices']
common=np.isfinite(v[5,:,:,:,:10]).all(axis=(1,2,3))
eligible=np.isfinite(fixed).all(axis=(3,4))&(doses>=0).all(3)&(doses==doses[:,:,:,0,None]).all(3)
assert np.all((lengths==lengths[:,:,:,0,None]).all(3)[eligible])
rows=[];coverage=[]
for group in ['pooled']+FAMILIES:
    fi=slice(None) if group=='pooled' else slice(FAMILIES.index(group),FAMILIES.index(group)+1)
    keep=eligible[:,fi].copy();fx=fixed[:,fi].copy()
    if group=='pooled':keep[~common]=False
    fx[~keep]=np.nan
    coverage.append(dict(group=group,documents=int(keep.any(axis=(1,2)).sum()),generations=int(keep.sum()),
        native_editable_generations=int((keep&(budgets[:,fi]>0)).sum()),
        positive_evaluator_dose_generations=int((keep&(doses[:,fi,: ,0]>0)).sum()),
        zero_evaluator_dose_generations=int((keep&(doses[:,fi,:,0]==0)).sum())))
    for pi,policy in enumerate(POLICIES[3:],3):
        delta=fx[:,:,:,pi]-fx[:,:,:,:3].mean(3)
        for ki,name in enumerate(['mean_nll_gain','changed_position_gain','unchanged_position_gain']):
            rows.append(dict(group=group,policy=policy,component=name,**bootstrap(delta[:,:,:,ki],draws)))
np.savez_compressed(OUT/'equal_evaluator_dose_arrays.npz',doses=doses,lengths=lengths,eligible=eligible)
result=dict(protocol=protocol,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),coverage=coverage,contrasts=rows,
    caveat='Output-dependent descriptive subset; excludes many native-editable cases; cannot identify causal effect of rarity or replace primary cohort.')
(OUT/'equal_evaluator_dose_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
