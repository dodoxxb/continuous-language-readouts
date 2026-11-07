"""Frozen-sequence readout study. Fits never use test scores for selection."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/confirmatory_readout_20261013'
P=json.loads((OUT/'protocol.json').read_text())

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def arrays_to_cache(directory,x,y,valid,ids=None):
    directory.mkdir(parents=True,exist_ok=True)
    for name,arr in [('x',x),('y',y),('valid',valid)]:np.save(directory/(name+'.npy'),np.asarray(arr))
    if ids is not None:np.save(directory/'sequence_ids.npy',np.asarray(ids))

def prepare_elf():
    target=OUT/'data/elf';target.mkdir(parents=True,exist_ok=True)
    if (target/'ready.json').exists():return
    shards=sorted((ROOT/P['elf_source']).glob('shard*.pt'),key=lambda x:int(x.stem[5:]))
    labels=[];valid=[];sizes=[]
    for f in shards:
        d=torch.load(f,map_location='cpu',mmap=True,weights_only=False)
        labels.append(d['labels'].numpy());valid.append(d['valid'].numpy());sizes.append(len(d['labels']))
    labels=np.concatenate(labels);valid=np.concatenate(valid)
    # Group exact native sequences before splitting; select one representative.
    seen={};duplicates=[];empty=[];unique=[]
    for i in range(len(labels)):
        if not valid[i].any():empty.append(i);continue
        h=hashlib.sha256(labels[i].tobytes()).hexdigest()
        if h in seen:duplicates.append([seen[h],i])
        else:seen[h]=i;unique.append(i)
    perm=np.random.default_rng(P['split_seed']).permutation(unique)
    nt=max(P['elf_train_budgets']);nv=P['elf_validation_sequences'];ne=P['elf_fixed_test_sequences']
    assert len(perm)>=nt+nv+ne
    split={'train':perm[:nt],'validation':perm[nt:nt+nv],'test':perm[nt+nv:nt+nv+ne]}
    assert not (set(split['train'])&set(split['validation']) or set(split['train'])&set(split['test']) or set(split['validation'])&set(split['test']))
    dump(target/'split.json',{'split_seed':P['split_seed'],'sequence_ids':{k:v.tolist() for k,v in split.items()},'source_sequences':len(labels),'unique_nonempty_native_sequences':len(unique),'dropped_empty_ids':empty,'duplicate_native_sequence_pairs':duplicates,'selection':'one representative of each exact native token sequence; partition before fitting','historical_data_reused':True})
    starts=np.cumsum([0]+sizes)
    # Save full precision features; no label-dependent feature selection.
    train_counts=valid[split['train']].sum(1);offsets=np.r_[0,np.cumsum(train_counts)]
    train=target/'train';train.mkdir(exist_ok=True)
    tx=np.lib.format.open_memmap(train/'x.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1]),512))
    ty=np.lib.format.open_memmap(train/'y.npy',mode='w+',dtype=np.int64,shape=(int(offsets[-1]),))
    np.save(train/'offsets.npy',offsets);np.save(train/'sequence_ids.npy',split['train'])
    maps={}
    for name in ['validation','test']:
        dest=target/name;dest.mkdir(exist_ok=True)
        maps[name]=np.lib.format.open_memmap(dest/'x.npy',mode='w+',dtype=np.float32,shape=(len(split[name]),labels.shape[1],512))
        np.save(dest/'y.npy',labels[split[name]]);np.save(dest/'valid.npy',valid[split[name]]);np.save(dest/'sequence_ids.npy',split[name])
    for si,f in enumerate(shards):
        d=torch.load(f,map_location='cpu',mmap=True,weights_only=False)
        feat=d['final_z'].numpy()
        for name,seqs in split.items():
            positions=np.flatnonzero((seqs>=starts[si])&(seqs<starts[si+1]))
            positions=positions[np.argsort(seqs[positions])]
            for j in positions:
                sid=seqs[j];local=sid-starts[si]
                if name=='train':
                    tx[offsets[j]:offsets[j+1]]=feat[local][valid[sid]]
                    ty[offsets[j]:offsets[j+1]]=labels[sid][valid[sid]]
                else:maps[name][j]=feat[local]
        print('prepared shard',si,flush=True)
    tx.flush();ty.flush()
    for m in maps.values():m.flush()
    budgets={str(n):int(offsets[n]) for n in P['elf_train_budgets']}
    manifest={'family':'elf','train_sequences':nt,'train_tokens':int(offsets[-1]),'budget_token_counts':budgets,'dim':512,'vocab':32100,'pad_id':0,'eos_id':1,'mask':'native non-padding positions; EOS already excluded by saved native decode','train_feature_sha256':digest(train/'x.npy'),'split_sha256':digest(target/'split.json'),'source_shards':[{'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size} for f in shards]}
    dump(target/'ready.json',manifest);print('ready',budgets,flush=True)

def train_cache_from_sequences(directory):
    x=np.load(directory/'train/x.npy',mmap_mode='r');y=np.load(directory/'train/y.npy');v=np.load(directory/'train/valid.npy')
    counts=v.sum(1);off=np.r_[0,np.cumsum(counts)]
    dest=directory/'train_flat';dest.mkdir(exist_ok=True)
    tx=np.lib.format.open_memmap(dest/'x.npy',mode='w+',dtype=np.float32,shape=(int(off[-1]),x.shape[-1]))
    ty=np.lib.format.open_memmap(dest/'y.npy',mode='w+',dtype=np.int64,shape=(int(off[-1]),))
    for i in range(len(x)):
        tx[off[i]:off[i+1]]=x[i][v[i]];ty[off[i]:off[i+1]]=y[i][v[i]]
    tx.flush();ty.flush();np.save(dest/'offsets.npy',off)

@torch.no_grad()
def evaluate(model,directory,device,save=None,eos_id=None,pad_id=0):
    x=np.load(directory/'x.npy',mmap_mode='r');y=np.load(directory/'y.npy');v=np.load(directory/'valid.npy')
    n,L,D=x.shape;pred=np.empty((n,L),dtype=np.int32)
    for i in range(n):
        for j in range(0,L,1024):
            z=torch.from_numpy(np.array(x[i,j:j+1024],copy=True)).to(device)
            # Evaluate the learned affine map in float32, without mixed-precision ties.
            pred[i,j:j+1024]=model(z.float()).argmax(-1).cpu().numpy()
    correct=((pred==y)&v).sum(1);den=v.sum(1)
    if eos_id is not None:
        kept=(pred==eos_id).cumsum(1)==0
        autonomous=np.where(kept,pred,pad_id)
        ac=((autonomous==y)&v).sum(1)
    else:ac=correct.copy()
    result={'agreement':float(correct.sum()/den.sum()),'macro_agreement':float(np.mean(correct[den>0]/den[den>0])),'autonomous_eos_agreement':float(ac.sum()/den.sum()),'sequences':n,'valid_tokens':int(den.sum()),'zero_valid_sequences':int((den==0).sum())}
    if save:
        save.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(save,pred=pred,correct=correct,denominator=den,autonomous_correct=ac)
    return result

def fit(family,budget,seed,device):
    data=OUT/'data'/family;manifest=json.loads((data/'ready.json').read_text())
    dest=OUT/'fits'/family/f'n{budget}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return
    flat=data/('train' if family=='elf' else 'train_flat')
    offsets=np.load(flat/'offsets.npy');n=int(offsets[budget])
    x=np.load(flat/'x.npy',mmap_mode='r');y=np.load(flat/'y.npy',mmap_mode='r')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed)
    model=nn.Linear(manifest['dim'],manifest['vocab']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=P['learning_rate'],weight_decay=P['weight_decay'])
    best=-1.;best_step=None;records=[];started=time.time()
    dump(dest/'configuration.json',{'family':family,'budget_sequences':budget,'train_tokens':n,'fit_seed':seed,'protocol_sha256':digest(OUT/'protocol.json'),'split_sha256':manifest.get('split_sha256'),'code_sha256':digest(Path(__file__)),'torch':torch.__version__,'training_dtype':'float32 parameters, bfloat16 autocast matmul','evaluation_dtype':'float32','fixed_updates':P['train_updates']})
    for step in range(1,P['train_updates']+1):
        idx=rng.integers(n,size=P['batch_tokens'])
        xb=torch.from_numpy(np.array(x[idx],copy=True)).to(device);yb=torch.from_numpy(np.array(y[idx],copy=True)).to(device)
        xb=xb+torch.randn_like(xb)*P['noise_aug']*xb.std().detach().clamp_min(1e-4)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(model(xb),yb)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%512==0:print(f'{family} n={budget} seed={seed} step={step} loss={loss.item():.4f} elapsed={time.time()-started:.1f}',flush=True)
        if step in P['validation_checkpoints']:
            model.eval();val=evaluate(model,data/'validation',device,eos_id=manifest.get('eos_id'),pad_id=manifest.get('pad_id',0));model.train()
            records.append({'step':step,'validation':val})
            if val['agreement']>best:
                best=val['agreement'];best_step=step
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},dest/'selected.pt')
            dump(dest/'validation_history.json',records)
    # The test is opened only after checkpoint selection is complete.
    model.load_state_dict(torch.load(dest/'selected.pt',weights_only=True,map_location=device));model.eval()
    result={'family':family,'train_sequences':budget,'train_tokens':n,'fit_seed':seed,'selected_update':best_step,'validation_agreement':best,'selection':'validation only','runtime_seconds':time.time()-started,'checkpoint_sha256':digest(dest/'selected.pt')}
    for name in ['test','fresh_test']:
        if (data/name/'x.npy').exists():result[name]=evaluate(model,data/name,device,save=dest/(name+'.npz'),eos_id=manifest.get('eos_id'),pad_id=manifest.get('pad_id',0))
    dump(dest/'selected.json',result);print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','fit','test-fresh']);ap.add_argument('--family',default='elf');ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    if args.mode=='prepare':prepare_elf();return
    budgets=P['elf_train_budgets'] if args.family=='elf' else [P['langflow_train_sequences']]
    stream=torch.cuda.Stream(device=args.device,priority=1)
    with torch.cuda.stream(stream):
        for budget in budgets:
            for seed in P['fit_seeds']:
                if args.mode=='fit':fit(args.family,budget,seed,args.device)
                else:
                    data=OUT/'data'/args.family;mf=json.loads((data/'ready.json').read_text());dest=OUT/'fits'/args.family/f'n{budget}_seed{seed}'
                    r=json.loads((dest/'selected.json').read_text())
                    if 'fresh_test' in r:continue
                    model=nn.Linear(mf['dim'],mf['vocab']).to(args.device)
                    model.load_state_dict(torch.load(dest/'selected.pt',map_location=args.device,weights_only=True));model.eval()
                    r['fresh_test']=evaluate(model,data/'fresh_test',args.device,save=dest/'fresh_test.npz',eos_id=mf.get('eos_id'),pad_id=mf.get('pad_id',0));dump(dest/'selected.json',r)

if __name__=='__main__':main()
