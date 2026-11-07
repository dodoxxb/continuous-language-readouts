"""Post-fit integrity and vocabulary-coverage diagnostics; no fit selection."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from confirmatory_readout_20261013 import OUT,P,dump
from confirmatory_analyze_20261013 import csvwrite

def hashes(y):
    return [hashlib.sha256(np.asarray(row).tobytes()).hexdigest() for row in y]

rows=[];audit={}
for family in ['elf','langflow']:
    root=OUT/'data'/family
    split=json.loads((root/'split.json').read_text())
    y=np.load(root/'test/y.npy');train=np.load(root/('train/y.npy' if family=='elf' else 'train_flat/y.npy'))
    counts=np.bincount(train,minlength=32128 if family=='elf' else 30522)
    if family=='langflow':
        allhash={n:hashes(np.load(root/n/'y.npy')) for n in ['train','validation','test']}
        audit[family]={'unique_sequences':{n:len(set(h)) for n,h in allhash.items()},'exact_native_sequence_overlap':{f'{a}--{b}':len(set(allhash[a])&set(allhash[b])) for a,b in [('train','validation'),('train','test'),('validation','test')]}}
        for seed in P['fit_seeds']:
            pred=np.load(OUT/'fits/langflow'/f'n1024_seed{seed}'/'test.npz')['pred']
            for name,m in [('all',np.ones_like(y,dtype=bool)),('unseen_training_token',counts[y]==0),('train_count_1_to_99',(counts[y]>0)&(counts[y]<100)),('train_count_at_least_100',counts[y]>=100)]:
                c=int(((pred==y)&m).sum());d=int(m.sum())
                rows.append({'family':family,'fit_seed':seed,'group':name,'correct':c,'tokens':d,'agreement':c/d if d else None})
    else:
        # Exact token sequence check only; fresh noise/seed provenance is separate.
        fresh=hashes(np.load(root/'fresh_test/y.npy'))
        hist=[]
        import torch
        source=Path(__file__).resolve().parents[1]/P['elf_source']
        for file in source.glob('shard*.pt'):
            hist.extend(hashes(torch.load(file,map_location='cpu',mmap=True,weights_only=False)['labels'].numpy()))
        audit[family]={'fresh_sequences':len(fresh),'unique_fresh_native_sequences':len(set(fresh)),'fresh_exact_native_sequence_overlap_with_historical_pool':len(set(fresh)&set(hist))}
audit['scope']='Post-fit data-integrity and LangFlow token-coverage diagnostics; no model or protocol changes.'
csvwrite(OUT/'langflow_coverage.csv',rows);dump(OUT/'data_integrity.json',audit)
print(json.dumps(audit,indent=2))
